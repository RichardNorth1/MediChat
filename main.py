from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from chat import handle_chat  # Import the handle_chat function from chat.py

app = FastAPI()

@app.get("/")
async def get():
    return {"message": "Welcome to the Chatbot API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = handle_chat(data)  # Use the handle_chat function from chat.py
        await websocket.send_text(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)