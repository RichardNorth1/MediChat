from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    """Render the main page where the doctor can view chats and send messages."""
    return render_template('doctor_chat.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
