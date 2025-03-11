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

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.patient_connections: Dict[str, WebSocket] = {}
        self.chatbot_connections: Dict[str, WebSocket] = {} 
        self.escalated_chats: List[str] = [] 
        self.lobby: List[str] = [] 
        self.assigned_chats: Dict[str, str] = {}

    async def connect_doctor(self, doctor_id: str, websocket: WebSocket):
        """ Adds a doctor to active connections """
        await websocket.accept()
        self.active_connections[doctor_id] = websocket
        logging.info(f"Doctor {doctor_id} connected.")

    async def connect_patient(self, patient_id: str, websocket: WebSocket):
        """ Adds a patient to the lobby or active connections """
        await websocket.accept()
        self.patient_connections[patient_id] = websocket
        self.lobby.append(patient_id) 
        logging.info(f"Patient {patient_id} connected and added to the lobby.")
        
        for doctor_id, doctor_websocket in self.active_connections.items():
            await doctor_websocket.send_text(json.dumps({
                "type": "lobby_update",
                "lobby": self.get_waiting_patients()
            }))

    async def connect_chatbot(self, client_id: str, websocket: WebSocket):
        """ Adds a chatbot connection """
        await websocket.accept()
        self.chatbot_connections[client_id] = websocket
        logging.info(f"Chatbot {client_id} connected.")

    async def assign_patient_to_doctor(self, doctor_id: str, patient_id: str):
        """ Assigns a patient to a doctor """
        if doctor_id not in self.active_connections:
            logging.error(f"Doctor {doctor_id} not connected.")
            return None
        
        if patient_id not in self.lobby:
            logging.error(f"Patient {patient_id} not in the lobby.")
            return None

        self.lobby.remove(patient_id)
        self.assigned_chats[doctor_id] = patient_id
        logging.info(f"Doctor {doctor_id} assigned to Patient {patient_id}")

        await self.active_connections[doctor_id].send_text(json.dumps({
            "type": "assigned",
            "doctor_id": doctor_id,
            "patient_id": patient_id
        }))

        for doc_id, doctor_websocket in self.active_connections.items():
            await doctor_websocket.send_text(json.dumps({
                "type": "lobby_update",
                "lobby": self.get_waiting_patients()
            }))

        return patient_id

    async def claim_escalated_chat(self, doctor_id: str, patient_id: str):
        """ Claims an escalated chat for a doctor and removes it from the escalated list """
        if patient_id in self.escalated_chats:
            self.escalated_chats.remove(patient_id)
            self.assigned_chats[doctor_id] = patient_id
            logging.info(f"Doctor {doctor_id} claimed escalated chat with Patient {patient_id}")

            for doc_id, doctor_websocket in self.active_connections.items():
                await doctor_websocket.send_text(json.dumps({
                    "type": "lobby_update",
                    "lobby": self.get_waiting_patients(),
                    "escalated_chats": self.get_escalated_chats()
                }))

            return True
        return False

    async def disconnect_doctor(self, doctor_id: str):
        """ Removes a doctor from active connections """
        if doctor_id in self.active_connections:
            if self.active_connections[doctor_id].application_state == WebSocketState.CONNECTED:
                await self.active_connections[doctor_id].close()
            del self.active_connections[doctor_id]
            logging.info(f"Doctor {doctor_id} disconnected.")
            
        if doctor_id in self.assigned_chats:
            del self.assigned_chats[doctor_id]

    async def disconnect_patient(self, patient_id: str):
        """ Removes a patient from connections and lobby """
        if patient_id in self.patient_connections:
            if self.patient_connections[patient_id].application_state == WebSocketState.CONNECTED:
                await self.patient_connections[patient_id].close()
            del self.patient_connections[patient_id]
            logging.info(f"Patient {patient_id} disconnected.")

        if patient_id in self.lobby:
            self.lobby.remove(patient_id)
        if patient_id in self.escalated_chats:
            self.escalated_chats.remove(patient_id)

    async def disconnect_chatbot(self, client_id: str):
        """ Disconnect the chatbot client """
        if client_id in self.chatbot_connections:
            if self.chatbot_connections[client_id].application_state == WebSocketState.CONNECTED:
                await self.chatbot_connections[client_id].close()
            del self.chatbot_connections[client_id]
            logging.info(f"Chatbot {client_id} disconnected.")

    async def escalate_chat(self, patient_id: str):
        """ Moves a patient to the escalated chats list """
        if patient_id not in self.escalated_chats:
            self.escalated_chats.append(patient_id)
            logging.info(f"Chat with Patient {patient_id} escalated.")
            for doctor_id, doctor_websocket in self.active_connections.items():
                await doctor_websocket.send_text(json.dumps({
                    "type": "lobby_update",
                    "lobby": self.get_waiting_patients()
                }))

    async def send_message(self, sender_id: str, receiver_id: str, message: str):
        """ Sends a message from a doctor or patient to the other """
        logging.info(f"Sending message from {sender_id} to {receiver_id}: {message}")
        if receiver_id in self.active_connections:
            await self.active_connections[receiver_id].send_text(message)
        elif receiver_id in self.patient_connections:
            await self.patient_connections[receiver_id].send_text(message)
        else:
            logging.error(f"Receiver {receiver_id} not connected.")

    def get_waiting_patients(self):
        """ Get list of patients in the lobby """ 
        return [{"patient_id": patient_id} for patient_id in self.lobby]

    def get_escalated_chats(self):
        """ Get list of patients in the escalated chats """
        return [{"client_id": patient_id} for patient_id in self.escalated_chats]


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
    await manager.connect_doctor(doctor_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            json_data = json.loads(data)
            # Assign patient to doctor
            if json_data.get("type") == "claim":
                patient_id = json_data.get("patient_id")
                logging.info(f"Doctor {doctor_id} is claiming patient {patient_id}")
                await manager.assign_patient_to_doctor(doctor_id, patient_id)
            # Send typing status to patient
            elif json_data.get("status") == "typing":
                patient_id = manager.assigned_chats.get(doctor_id)
                if patient_id and patient_id in manager.patient_connections:
                    logging.info(f"Doctor {doctor_id} is typing to patient {patient_id}")
                    await manager.send_message(doctor_id, patient_id, json.dumps({
                        "status": "typing",
                        "sender": "doctor"
                    }))
            # If not typing status or assignment, forward message to patient
            else:
                patient_id = json_data.get("patient_id")
                logging.info(f"Forwarding message to patient {patient_id}")
                if patient_id and patient_id in manager.patient_connections:
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
            data = await websocket.receive_text()
            json_data = json.loads(data)

            if json_data.get("status") == "typing":
                # Send patient is typing message to doctor
                doctor_id = next((doc for doc, pat in manager.assigned_chats.items() if pat == patient_id), None)
                if doctor_id and doctor_id in manager.active_connections:
                    logging.info(f"Patient {patient_id} is typing to doctor {doctor_id}")
                    await manager.send_message(patient_id, doctor_id, json.dumps({
                        "status": "typing",
                        "sender": "patient",
                        "patient_id": patient_id
                    }))
            else:
                # Forward messages from patient to doctor
                doctor_id = next((doc for doc, pat in manager.assigned_chats.items() if pat == patient_id), None)
                logging.info(f"Forwarding message from patient {patient_id} to doctor {doctor_id}")
                if doctor_id and doctor_id in manager.active_connections:
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
            data = await websocket.receive_text()
            if not data:
                continue

            try:
                data_json = json.loads(data)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON: {data}")
                continue

            # Process the typing status from the client
            if data_json.get("status") == "typing":
                sender = data_json.get("sender", "patient")
                logging.info(f"Patient {client_id} is typing")
                await websocket.send_text(json.dumps({"status": "typing", "sender": sender}))
                continue
            if data_json.get("status") == "form_data":
                # Send the form data to the doctor
                await manager.broadcast_to_doctors(data_json.get("message", ""))
                continue
            else:
                # Handle the chatbot's response
                typing_message = json.dumps({"status": "typing", "sender": "chatbot"})
                await websocket.send_text(typing_message)
                response = handle_chat(data_json.get("message", ""))
                response_message = json.dumps({"message": response["message"], "sender": "chatbot"})
                await websocket.send_text(response_message)
                logging.info(f"Sent to user {client_id}: {response_message}")
                
                # Handle chat escalation
                if response.get("escalate"):
                    await manager.escalate_chat(client_id)
                    escalation_message = {"status": "escalated", "client_id": client_id}
                    await websocket.send_text(json.dumps(escalation_message))

    except WebSocketDisconnect:
        logging.info(f"Patient {client_id} disconnected due to WebSocketDisconnect")
        await manager.disconnect_chatbot(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)