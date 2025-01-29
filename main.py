from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from src.chat_v2 import handle_chat
import logging
import json

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS settings to allow the Flask app to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ConnectionManager tracks active connections and escalated chats
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.client_statuses: Dict[str, str] = {}  # Track client statuses
        self.escalated_chats: List[str] = []  # Track escalated chats

    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a WebSocket to a specific client ID."""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        self.client_statuses[client_id] = "connected"
        logging.info(f"Client {client_id} connected")

    def disconnect(self, websocket: WebSocket, client_id: str):
        """Disconnect a WebSocket from a specific client ID."""
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
                del self.client_statuses[client_id]
            logging.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(message)
            logging.info(f"Sent personal message: {message}")

    async def broadcast_to_doctors(self, message: str):
        """Broadcast a message to all doctor WebSocket connections."""
        doctor_connections = self.active_connections.get("-1", [])
        for connection in doctor_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_text(json.dumps({"message": message}))
        logging.info(f"Broadcasted to doctors: {message}")

    async def escalate_chat(self, client_id: str):
        """Mark a chat as escalated and notify doctors."""
        if client_id not in self.escalated_chats:
            self.escalated_chats.append(client_id)

    async def send_typing_status(self, client_id: str, sender: str):
        """Send typing status to the specified client."""
        message = json.dumps({"status": "typing", "sender": sender})
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)

# Instantiate the connection manager
connection_manager = ConnectionManager()

@app.get("/chats")
async def get_chats():
    """Return the statuses of all clients and their escalated chats."""
    return {"client_statuses": connection_manager.client_statuses, "escalated_chats": connection_manager.escalated_chats}

@app.get("/api/escalated_chats")
async def get_escalated_chats():
    """Return the list of escalated chats (client IDs)."""
    escalated_chats = [
        {"client_id": client_id} 
        for client_id in connection_manager.escalated_chats
    ]
    return escalated_chats

@app.websocket("/ws/doctor/{client_id}")
async def doctor_patient_endpoint(websocket: WebSocket, client_id: str):
    await connection_manager.connect(websocket, client_id)
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

            logging.debug(f"Received from doctor for patient {client_id}: {data_json}")

            if data_json.get("status") == "typing":
                sender = data_json.get("sender", "doctor")
                await connection_manager.send_typing_status(client_id, sender)
                continue

            patient_connections = connection_manager.active_connections.get(client_id)
            if patient_connections:
                message = {"message": data_json.get("message", ""), "sender": "doctor"}
                for patient_connection in patient_connections:
                    if patient_connection != websocket:  # Exclude the sender
                        await connection_manager.send_personal_message(json.dumps(message), patient_connection)
                logging.debug(f"Sent message from doctor to patient #{client_id}: {message}")
            else:
                logging.debug(f"No active connection found for patient #{client_id}")

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, client_id)
        logging.debug(f"Doctor disconnected from patient #{client_id}")


@app.websocket("/ws/chatbot/{client_id}")
async def chatbot_endpoint(websocket: WebSocket, client_id: str):
    await connection_manager.connect(websocket, client_id)
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

            logging.info(f"Received from user {client_id}: {data_json}")

            if data_json.get("status") == "typing":
                sender = data_json.get("sender", "patient")
                logging.info(f"User {client_id} is typing")
                await connection_manager.send_typing_status(client_id, sender)
                continue

            # Indicate that the chatbot is typing
            typing_message = json.dumps({"status": "typing", "sender": "chatbot"})
            await websocket.send_text(typing_message)

            response = handle_chat(data_json.get("message", ""))
            if not response or "message" not in response:
                response = {"message": "No message received"}

            await websocket.send_text(json.dumps(response))
            logging.info(f"Sent to user {client_id}: {response}")

            if response.get("escalate"):
                connection_manager.client_statuses[client_id] = "escalated"
                await connection_manager.escalate_chat(client_id)
                escalation_message = {"status": "escalated", "client_id": client_id}
                await websocket.send_text(json.dumps(escalation_message))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, client_id)
        logging.info(f"User {client_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
