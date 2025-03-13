from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from pydantic import BaseModel
from chat import handle_chat
import logging
import json

class ClaimRequest(BaseModel):
    doctor_id: str
    patient_id: str

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ConnectionManager class to manage WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {} # Doctor connections
        self.patient_connections: Dict[str, WebSocket] = {} # Patient connections
        self.chatbot_connections: Dict[str, WebSocket] = {} # Chatbot connections
        self.lobby: List[str] = [] # Patients in the lobby
        self.assigned_chats: Dict[str, List[str]] = {} # Assigned chats

    async def connect_doctor(self, doctor_id: str, websocket: WebSocket):
        """ Adds a doctor to active connections """
        await websocket.accept()
        self.active_connections[doctor_id] = websocket
        logging.info(f"Doctor {doctor_id} connected.")

    async def connect_patient(self, patient_id: str, websocket: WebSocket):
        """ Adds a patient to the lobby or active connections """
        await websocket.accept()
        self.patient_connections[patient_id] = websocket
        logging.info(f"Patient {patient_id} connected and added to the lobby.")
        await self.broadcast_lobby_update()

    async def connect_chatbot(self, client_id: str, websocket: WebSocket):
        """ Adds a chatbot connection """
        await websocket.accept()
        self.chatbot_connections[client_id] = websocket
        logging.info(f"Chatbot {client_id} connected.")

    async def assign_patient(self, doctor_id: str, patient_id: str):
        """Assigns a patient to a doctor from the lobby."""
        logging.info("Assigning patient")
        # Check if doctor is connected
        if doctor_id not in self.active_connections:
            logging.error(f"Doctor {doctor_id} not connected.")
            return {"success": False, "message": "Doctor not connected."}
        
        # Check if patient is in the lobby
        if patient_id in self.lobby:
            self.lobby.remove(patient_id)
            logging.info(f"Patient {patient_id} removed from lobby.")
        else:
            logging.error(f"Patient {patient_id} not found in lobby.")
            return {"success": False, "message": "Patient not found in lobby."}
        
        # if doctor is not in assigned chats, add them
        if doctor_id not in self.assigned_chats:
            self.assigned_chats[doctor_id] = []

        # Assign the patient to the doctor
        self.assigned_chats[doctor_id].append(patient_id)
        logging.info(f"Doctor {doctor_id} assigned to Patient {patient_id}.")

        # Notify the doctor of the assignment
        await self.active_connections[doctor_id].send_text(json.dumps({
            "type": "assigned",
            "doctor_id": doctor_id,
            "assigned_patients": self.assigned_chats[doctor_id]
        }))

        # Broadcast the updated lobby list
        await self.broadcast_lobby_update()

        return {"success": True, "message": "Patient assigned successfully."}

    async def broadcast_lobby_update(self):
        """ Broadcast the updated lobby list to all connected doctors """
        lobby_update_message = json.dumps({
            "type": "lobby_update",
            "lobby": self.get_waiting_patients()
        })
        await self.broadcast_to_doctors(lobby_update_message)

    async def broadcast_to_doctors(self, message: str):
        """ Broadcast a message to all connected doctors """
        for doctor_websocket in self.active_connections.values():
            await doctor_websocket.send_text(message)

    async def disconnect_doctor(self, doctor_id: str):
        """ Removes a doctor from active connections """
        # if doctor is connected
        if doctor_id in self.active_connections:
            # if doctor is connected, close the connection
            if self.active_connections[doctor_id].application_state == WebSocketState.CONNECTED:
                await self.active_connections[doctor_id].close()
            # remove the doctor from active connections
            del self.active_connections[doctor_id]
            logging.info(f"Doctor {doctor_id} disconnected.")
        # remove the doctor from assigned chats
        if doctor_id in self.assigned_chats:
            del self.assigned_chats[doctor_id]

    async def disconnect_patient(self, patient_id: str):
        """ Removes a patient from connections and lobby """
        # if patient in patient connections
        if patient_id in self.patient_connections:
            # if patient is connected, close the connection
            if self.patient_connections[patient_id].application_state == WebSocketState.CONNECTED:
                await self.patient_connections[patient_id].close()
            # remove the patient from patient connections
            del self.patient_connections[patient_id]
            logging.info(f"Patient {patient_id} disconnected.")
        # remove the patient from the lobby
        if patient_id in self.lobby:
            self.lobby.remove(patient_id)

    async def disconnect_chatbot(self, client_id: str):
        """ Disconnect the chatbot client """
        # if client in chatbot connections
        if client_id in self.chatbot_connections:
            # if client is connected, close the connection
            if self.chatbot_connections[client_id].application_state == WebSocketState.CONNECTED:
                await self.chatbot_connections[client_id].close()
            # remove the client from chatbot connections
            del self.chatbot_connections[client_id]
            logging.info(f"Chatbot {client_id} disconnected.")

    async def escalate_chat(self, patient_id: str):
        """ Moves a patient to the lobby """
        # if patient is not in lobby
        if patient_id not in self.lobby:
            # add patient to lobby
            self.lobby.append(patient_id)
            logging.info(f"Chat with Patient {patient_id} escalated.")
            # broadcast the updated lobby list
            await self.broadcast_lobby_update()


    async def send_message(self, sender_id: str, receiver_id: str, message: str):
        """ Sends a message from a doctor to a patient or vice versa. """
        logging.info(f"Sending message from {sender_id} to {receiver_id}: {message}")
        # if sender is in active connections and assigned chats
        if sender_id in self.active_connections and sender_id in self.assigned_chats:
            # if receiver is in assigned chats
            if receiver_id in self.assigned_chats[sender_id]:
                # if receiver is in patient connections
                if receiver_id in self.patient_connections:
                    # send the message to the receiver
                    await self.patient_connections[receiver_id].send_text(message)
                else:
                    logging.error(f"Patient {receiver_id} not connected.")
            else:
                logging.error(f"Doctor {sender_id} is not assigned to Patient {receiver_id}.")
        # if sender is in patient connections
        elif sender_id in self.patient_connections:
            # find the dooctor ID
            doctor_id = next((doc for doc, patients in self.assigned_chats.items() if sender_id in patients), None)
            # if doctor ID is found and doctor is in active connections
            if doctor_id and doctor_id in self.active_connections:
                # send the message to the doctor
                await self.active_connections[doctor_id].send_text(message)
            else:
                logging.error(f"Doctor for Patient {sender_id} not connected.")

    def get_waiting_patients(self):
        """ Get list of patients in the lobby """ 
        return [{"patient_id": patient_id} for patient_id in self.lobby]

    def get_doctors_assigned_patients(self, doctor_id: str):
        """ Get the list of patients assigned to a doctor """
        return self.assigned_chats.get(doctor_id, [])

# Create a ConnectionManager instance
manager = ConnectionManager()

@app.get("/api/assigned_patients/{doctor_id}")
async def get_assigned_patients(doctor_id: str):
    """Return the list of patients assigned to a doctor."""
    print(manager.get_doctors_assigned_patients(doctor_id))
    return {"patients": manager.get_doctors_assigned_patients(doctor_id)}

@app.get("/api/lobby/")
async def get_lobby():
    """Return the list of waiting patients."""
    return {"lobby": manager.get_waiting_patients()}

@app.post("/api/claim_patient")
async def claim_patient(request: ClaimRequest):
    """Assign a patient to a doctor with detailed error handling."""
    logging.info("Assigning patient via API")
    success = await manager.assign_patient(request.doctor_id, request.patient_id)
    if not success:
        return {"success": False, "message": "Patient claim unsuccessful"}
    return {"success": True, "message": "Patient claimed successfully"}

@app.websocket("/ws/doctor/{doctor_id}")
async def websocket_doctor(websocket: WebSocket, doctor_id: str):
    await manager.connect_doctor(doctor_id, websocket)
    try:
        while True:
            # recieve messages from the doctor
            data = await websocket.receive_text()
            #parse the message
            json_data = json.loads(data)
            # if the message is a typing status
            if json_data.get("status") == "typing":
                # get the patient ID
                patient_id = json_data.get("patient_id")
                # if the patient is in the assigned chats
                if patient_id in manager.assigned_chats.get(doctor_id, []):
                    # send a typing status to the patient
                    await manager.send_message(doctor_id, patient_id, json.dumps({
                        "status": "typing",
                        "sender": "doctor"
                    }))
            # if the message is not a typing status e.g. a message
            else:
                # get the patient ID
                patient_id = json_data.get("patient_id")
                # if the patient is in the assigned chats
                if patient_id in manager.assigned_chats.get(doctor_id, []):
                    # send the message to the patient
                    await manager.send_message(doctor_id, patient_id, json.dumps({
                        "message": json_data.get("message"),
                        "sender": "doctor"
                    }))
    except WebSocketDisconnect:
        logging.info(f"Doctor {doctor_id} disconnected")
        await manager.disconnect_doctor(doctor_id)

@app.websocket("/ws/patient/{patient_id}")
async def websocket_patient(websocket: WebSocket, patient_id: str):
    await manager.connect_patient(patient_id, websocket)
    try:
        while True:
            # recieve messages from the patient
            data = await websocket.receive_text()
            #parse the message
            json_data = json.loads(data)
            # if the message is a typing status
            if json_data.get("status") == "typing":
                # find the doctor ID
                doctor_id = next((doc for doc, patients in manager.assigned_chats.items() if patient_id in patients), None)
                # if doctor ID is found and doctor is in active connections
                if doctor_id and doctor_id in manager.active_connections:
                    # send a typing status to the doctor
                    await manager.send_message(patient_id, doctor_id, json.dumps({
                        "status": "typing",
                        "sender": "patient",
                        "patient_id": patient_id
                    }))
            # if the message is not a typing status e.g. a message
            else:
                # find the doctor ID
                doctor_id = next((doc for doc, patients in manager.assigned_chats.items() if patient_id in patients), None)
                # if doctor ID is found and doctor is in active connections
                if doctor_id and doctor_id in manager.active_connections:
                    # send the message to the doctor
                    await manager.send_message(patient_id, doctor_id, json.dumps({
                        "message": json_data.get("message"),
                        "patient_id": patient_id
                    }))
    except WebSocketDisconnect:
        logging.info(f"Patient {patient_id} disconnected")
        await manager.disconnect_patient(patient_id)

@app.websocket("/ws/chatbot/{client_id}")
async def chatbot_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect_chatbot(client_id, websocket)
    try:
        while True:
            # recieve messages from the patient
            data = await websocket.receive_text()
            #parse the message
            data_json = json.loads(data)
            # if the message is a typing status
            if data_json.get("status") == "typing":
                # get the sender
                sender = data_json.get("sender", "patient")
                logging.info(f"Patient {client_id} is typing")
                # send a typing status from the patient to the chatbot
                await websocket.send_text(json.dumps({"status": "typing", "sender": sender}))
                continue
            # if the message is form data type
            if data_json.get("status") == "form_data":
                #broadcast the form data to all doctors
                await manager.broadcast_to_doctors(data_json.get("message", ""))
                continue

            # if the message is a message
            else:
                # send a typing status to the user
                typing_message = json.dumps({"status": "typing", "sender": "chatbot"})
                # send the typing status to the user
                await websocket.send_text(typing_message)
                # use the handle chat function to get a response
                response = handle_chat(data_json.get("message", ""))
                #transform the data to json
                response_message = json.dumps({"message": response["message"], "sender": "chatbot"})
                # send the response to the user
                await websocket.send_text(response_message)
                logging.info(f"Sent to user {client_id}: {response_message}")
                
                # if the response has an escalate key
                if response.get("escalate"):
                    # escalate the chat
                    await manager.escalate_chat(client_id)
                    # create escalation message for the frontend
                    escalation_message = {"status": "escalated", "client_id": client_id}
                    # send the escalation message to the frontend to switch active socket
                    await websocket.send_text(json.dumps(escalation_message))
    except WebSocketDisconnect:
        logging.info(f"Patient {client_id} disconnected due to WebSocketDisconnect")
        await manager.disconnect_chatbot(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)