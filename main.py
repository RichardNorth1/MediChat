from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from chat import handle_chat
import logging
import json

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
        self.escalated_chats: List[str] = [] # Escalated chats
        self.lobby: List[str] = [] # Patients in the lobby
        self.assigned_chats: Dict[str, str] = {} # Assigned chats (doctor_id: patient_id)

    async def connect_doctor(self, doctor_id: str, websocket: WebSocket):
        """ Adds a doctor to active connections """
        await websocket.accept()
        #add the doctor to the active connections
        self.active_connections[doctor_id] = websocket
        logging.info(f"Doctor {doctor_id} connected.")

    async def connect_patient(self, patient_id: str, websocket: WebSocket):
        """ Adds a patient to the lobby or active connections """
        await websocket.accept()
        # Add the patient to the patient connections
        self.patient_connections[patient_id] = websocket
        self.lobby.append(patient_id) 
        logging.info(f"Patient {patient_id} connected and added to the lobby.")
        # update lobby 
        lobby_update_message = json.dumps({
            "type": "lobby_update",
            "lobby": self.get_waiting_patients()
        })
        # Broadcast the updated lobby to all doctors
        await self.broadcast_to_doctors(lobby_update_message)

    async def connect_chatbot(self, client_id: str, websocket: WebSocket):
        """ Adds a chatbot connection """
        await websocket.accept()
        # Add the chatbo client to the chatbot connections
        self.chatbot_connections[client_id] = websocket
        logging.info(f"Chatbot {client_id} connected.")

    async def assign_patient_to_doctor(self, doctor_id: str, patient_id: str):
        """ Assigns a patient to a doctor """
        # check if the doctor is connected
        if doctor_id not in self.active_connections:
            logging.error(f"Doctor {doctor_id} not connected.")
            return None
        
        # check if the patient is in the lobby
        if patient_id not in self.lobby:
            logging.error(f"Patient {patient_id} not in the lobby.")
            return None
        # remove the patient from the lobby and assign to the doctor
        self.lobby.remove(patient_id)
        self.assigned_chats[doctor_id] = patient_id
        logging.info(f"Doctor {doctor_id} assigned to Patient {patient_id}")

        # send assigned message to the doctor
        await self.active_connections[doctor_id].send_text(json.dumps({
            "type": "assigned",
            "doctor_id": doctor_id,
            "patient_id": patient_id
        }))

        # send lobby update to all doctors
        for doc_id, doctor_websocket in self.active_connections.items():
            await doctor_websocket.send_text(json.dumps({
                "type": "lobby_update",
                "lobby": self.get_waiting_patients()
            }))

        return patient_id

    async def broadcast_to_doctors(self, message: str):
        """ Broadcast a message to all connected doctors """
        for doctor_websocket in self.active_connections.values():
            await doctor_websocket.send_text(message)

    async def claim_escalated_chat(self, doctor_id: str, patient_id: str):
        """ Claims an escalated chat for a doctor and removes it from the escalated list """
        # check if the patient is in the escalated chats
        if patient_id in self.escalated_chats:
            # remove the patient from the escalated chats and assign to the doctor
            self.escalated_chats.remove(patient_id)
            self.assigned_chats[doctor_id] = patient_id
            logging.info(f"Doctor {doctor_id} claimed escalated chat with Patient {patient_id}")

            # send assigned message to the doctors
            for doc_id, doctor_websocket in self.active_connections.items():
                await doctor_websocket.send_text(json.dumps({
                    "type": "lobby_update",
                    "lobby": self.get_waiting_patients(),
                    "escalated_chats": self.get_escalated_chats()
                }))
            #return the is successful
            return True
        return False

    async def disconnect_doctor(self, doctor_id: str):
        """ Removes a doctor from active connections """
        # check if the doctor is in the active connections
        if doctor_id in self.active_connections:
            # check if the doctor is connected
            if self.active_connections[doctor_id].application_state == WebSocketState.CONNECTED:
                # close the connection
                await self.active_connections[doctor_id].close()
            # remove the doctor from the active connections
            del self.active_connections[doctor_id]
            logging.info(f"Doctor {doctor_id} disconnected.")
        # check if the doctor is in the assigned chats
        if doctor_id in self.assigned_chats:
            # remove the doctor from the assigned chats
            del self.assigned_chats[doctor_id]

    async def disconnect_patient(self, patient_id: str):
        """ Removes a patient from connections and lobby """
        # check if the patient is in the patient connections
        if patient_id in self.patient_connections:
            # check if the patient is connected
            if self.patient_connections[patient_id].application_state == WebSocketState.CONNECTED:
                # close the connection
                await self.patient_connections[patient_id].close()
            # remove the patient from the patient connections
            del self.patient_connections[patient_id]
            logging.info(f"Patient {patient_id} disconnected.")

        # check if the patient is in the lobby
        if patient_id in self.lobby:
            # remove the patient from the lobby
            self.lobby.remove(patient_id)
        # check if the patient is in the escalated chats
        if patient_id in self.escalated_chats:
            # remove the patient from the escalated chats
            self.escalated_chats.remove(patient_id)

    async def disconnect_chatbot(self, client_id: str):
        """ Disconnect the chatbot client """
        # check if the chatbot client is in the chatbot connections
        if client_id in self.chatbot_connections:
            # check if the chatbot client is connected
            if self.chatbot_connections[client_id].application_state == WebSocketState.CONNECTED:
                # close the connection
                await self.chatbot_connections[client_id].close()
            # remove the chatbot client from the chatbot connections
            del self.chatbot_connections[client_id]
            logging.info(f"Chatbot {client_id} disconnected.")

    async def escalate_chat(self, patient_id: str):
        """ Moves a patient to the escalated chats list """
        # check if the patient is in the escalated chats
        if patient_id not in self.escalated_chats:
            # add the patient to the escalated chats
            self.escalated_chats.append(patient_id)
            logging.info(f"Chat with Patient {patient_id} escalated.")
            # send lobby update to all doctors
            for doctor_id, doctor_websocket in self.active_connections.items():
                await doctor_websocket.send_text(json.dumps({
                    "type": "lobby_update",
                    "lobby": self.get_waiting_patients()
                }))

    async def send_message(self, sender_id: str, receiver_id: str, message: str):
        """ Sends a message from a doctor or patient to the other """
        logging.info(f"Sending message from {sender_id} to {receiver_id}: {message}")
        # check if the receiver is in the doctor connections
        if receiver_id in self.active_connections:
            # send the message to the receiver
            await self.active_connections[receiver_id].send_text(message)
        # check if the receiver is in the patient connections
        elif receiver_id in self.patient_connections:
            # send the message to the receiver
            await self.patient_connections[receiver_id].send_text(message)
        else:
            logging.error(f"Receiver {receiver_id} not connected.")

    def get_waiting_patients(self):
        """ Get list of patients in the lobby """ 
        return [{"patient_id": patient_id} for patient_id in self.lobby]

    def get_escalated_chats(self):
        """ Get list of patients in the escalated chats """
        return [{"client_id": patient_id} for patient_id in self.escalated_chats]

# Create a ConnectionManager instance
manager = ConnectionManager()

@app.get("/api/escalated_chats")
async def get_escalated_chats():
    """Return the list of escalated chats (client IDs)."""
    return manager.get_escalated_chats()

@app.get("/api/lobby/")
async def get_lobby():
    """Return the list of waiting patients."""
    return {"lobby": manager.get_waiting_patients()}

@app.websocket("/ws/doctor/{doctor_id}")
async def websocket_doctor(websocket: WebSocket, doctor_id: str):
    # Connect doctor to WebSocket
    await manager.connect_doctor(doctor_id, websocket)
    try:
        while True:
            # Receive incoming messages from the doctor
            data = await websocket.receive_text()
            # Parse JSON data
            json_data = json.loads(data)
            # if the message is a claim message
            if json_data.get("type") == "claim":
                patient_id = json_data.get("patient_id")
                logging.info(f"Doctor {doctor_id} is claiming patient {patient_id}")
                # Claim the escalated chat
                await manager.assign_patient_to_doctor(doctor_id, patient_id)
            # if message is a typing status
            elif json_data.get("status") == "typing":
                # get the patient ID
                patient_id = manager.assigned_chats.get(doctor_id)
                # check if the patient is connected
                if patient_id and patient_id in manager.patient_connections:
                    logging.info(f"Doctor {doctor_id} is typing to patient {patient_id}")
                    # send the typing status to the patient
                    await manager.send_message(doctor_id, patient_id, json.dumps({
                        "status": "typing",
                        "sender": "doctor"
                    }))
            # If it is a message
            else:
                # get patient ID
                patient_id = json_data.get("patient_id")
                logging.info(f"Forwarding message to patient {patient_id}")
                # check if the patient is connected
                if patient_id and patient_id in manager.patient_connections:
                    # forward the message to the patient
                    await manager.send_message(doctor_id, patient_id, json.dumps({
                        "message": json_data.get("message"),
                        "sender": "doctor"
                    }))
    except WebSocketDisconnect:
        logging.info(f"Doctor {doctor_id} disconnected due to WebSocketDisconnect")
        await manager.disconnect_doctor(doctor_id)

@app.websocket("/ws/patient/{patient_id}")
async def websocket_patient(websocket: WebSocket, patient_id: str):
    await manager.connect_patient(patient_id, websocket)

    try:
        while True:
            # Receive incoming messages from the patient
            data = await websocket.receive_text()
            # Parse the data as JSON
            json_data = json.loads(data)

            # Check if the message is a typing status
            if json_data.get("status") == "typing":
                # find the doctor ID from the assigned chats
                doctor_id = next((doc for doc, pat in manager.assigned_chats.items() if pat == patient_id), None)
                # check if the doctor is connected
                if doctor_id and doctor_id in manager.active_connections:
                    logging.info(f"Patient {patient_id} is typing to doctor {doctor_id}")
                    # send the typing status to the doctor
                    await manager.send_message(patient_id, doctor_id, json.dumps({
                        "status": "typing",
                        "sender": "patient",
                        "patient_id": patient_id
                    }))
            else:
                # find the doctor ID from the assigned chats
                doctor_id = next((doc for doc, pat in manager.assigned_chats.items() if pat == patient_id), None)
                logging.info(f"Forwarding message from patient {patient_id} to doctor {doctor_id}")
                # check if the doctor is connected
                if doctor_id and doctor_id in manager.active_connections:
                    # forward the message to the doctor
                    await manager.send_message(patient_id, doctor_id, json.dumps({
                        "message": json_data.get("message"),
                        "sender": "patient"
                    }))
    except WebSocketDisconnect:
        logging.info(f"Patient {patient_id} disconnected due to WebSocketDisconnect")
        await manager.disconnect_patient(patient_id)

@app.websocket("/ws/chatbot/{client_id}")
async def chatbot_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for chatbot."""
    await manager.connect_chatbot(client_id, websocket)
    try:
        while True:
            # Receive incoming messages from the user
            data = await websocket.receive_text()
            # Parse the data as JSON
            data_json = json.loads(data)

            # check if the message is a typing status
            if data_json.get("status") == "typing":
                # Send typing status from the patient    
                sender = data_json.get("sender", "patient")
                logging.info(f"Patient {client_id} is typing")
                await websocket.send_text(json.dumps({"status": "typing", "sender": sender}))
                continue
            # check if the message is a form data
            if data_json.get("status") == "form_data":
                # Send the form data to the doctor
                await manager.broadcast_to_doctors(data_json.get("message", ""))
                continue
            else:
                # send the typing status from the chatbot to the user
                typing_message = json.dumps({"status": "typing", "sender": "chatbot"})
                await websocket.send_text(typing_message)
                # use the chat function to handle the message and get the response from the chatbot
                response = handle_chat(data_json.get("message", ""))
                # transform the response to JSON
                response_message = json.dumps({"message": response["message"], "sender": "chatbot"})
                # send the response to the user
                await websocket.send_text(response_message)
                logging.info(f"Sent to user {client_id}: {response_message}")
                
                # check if the response from the chatbot is an escalation message
                if response.get("escalate"):
                    # add the patient to the escalated chats list
                    await manager.escalate_chat(client_id)
                    escalation_message = {"status": "escalated", "client_id": client_id}
                    # send the escalation message to the frontend for the endpoint switch
                    await websocket.send_text(json.dumps(escalation_message))

    except WebSocketDisconnect:
        logging.info(f"Patient {client_id} disconnected due to WebSocketDisconnect")
        await manager.disconnect_chatbot(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)