from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

# Flask API endpoint to fetch the escalated chats from FastAPI
@app.route("/api/escalated_chats")
def get_escalated_chats():
    """Fetch a list of escalated chats from FastAPI."""
    fastapi_url = "http://localhost:8000/chats"
    response = requests.get(fastapi_url)
    if response.status_code == 200:
        data = response.json()
        return jsonify(data['escalated_chats'])
    return jsonify([])

# The main route renders the chat interface for doctors
@app.route("/")
def index():
    """Render the main page where the doctor can view chats and send messages."""
    return render_template('doctor_chat.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
