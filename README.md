# MediChat

MediChat is a web-based application designed to facilitate communication between patients and doctors. The application includes a chatbot for initial patient interaction and escalates the chat to a doctor when necessary. The project is built using FastAPI for the backend and WebSocket for real-time communication.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Endpoints](#endpoints)

## Features

- Real-time chat between patients and doctors using WebSocket.
- Initial interaction with a chatbot.
- Escalation of chat to a doctor when necessary.
- Typing indicators for both patients and doctors.
- Form submission for patient details.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/RichardNorth1/MediChat.git
    cd MediChat
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Note that the "language_tool_python" requires Java to run so please ensure you have Java SDK installed in order to run the project.

## Usage
1. activate virtual environment
    '''bash
    source venv/bin/activate  or venv\Scripts\activate on windows
    '''
2. Start the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

3. open new terminal and activate virtual environment as in step one

4. change working directories 
    ```bash
    cd doctorFrontend
    ```

5. run the doctor front end flask app
    ```bash
    python app.py  
    ```
6. open new terminal and activate virtual environment as in step one

7. change working directories 
    ```bash
    cd patientFrontend
    ```

8. run the doctor front end flask app
    ```bash
    python app.py  
    ```

## Project Structure
MediChat/ 
├── main.py 
├── chat.py 
├── README.md 
├── requirements.txt 
├── patientFrontend/ 
│   ├── templates/ 
│   ├── patient_chat.html 
│   └── static/ 
│   ├── styles.css 
├── doctorFrontend/ 
│   ├── templates/ 
│   │ ├── doctor_chat.html 
│   └── static/ 
│     ├── styles.css 


## Endpoints

### WebSocket Endpoints

- `/ws/doctor/{doctor_id}`: WebSocket endpoint for doctors.
- `/ws/patient/{patient_id}`: WebSocket endpoint for patients.
- `/ws/chatbot/{client_id}`: WebSocket endpoint for chatbot clients.

### REST Endpoints

- `GET /api/escalated_chats`: Returns the list of escalated chats (client IDs).
- `GET /api/lobby/`: Returns the list of waiting patients.
