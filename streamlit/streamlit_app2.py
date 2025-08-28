import streamlit as st
import requests
from supabase import create_client, Client
from uuid import UUID
from typing import Dict, List, Optional
import time
import os
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

# API and Supabase configuration
API_BASE_URL = os.getenv("API_BASE_URL")
SUPABASE_URL =   os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_poll_time" not in st.session_state:
    st.session_state.last_poll_time = 0
if "input_key" not in st.session_state:
    st.session_state.input_key = "user_input_0"
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0

def fetch_sessions() -> List[Dict]:
    """Fetch all sessions for the user."""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch sessions: {e}")
        return []

def fetch_messages(session_id: UUID) -> List[Dict]:
    """Fetch messages for a given session."""
    try:
        response = requests.get(f"{API_BASE_URL}/messages/{session_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch messages: {e}")
        return []

def create_session() -> Optional[Dict]:
    """Create a new session."""
    try:
        response = requests.post(f"{API_BASE_URL}/sessions")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to create session: {e}")
        return None

def send_message(content: str, session_id: Optional[UUID] = None) -> Optional[Dict]:
    """Send a message and get AI response."""
    try:
        payload = {"content": content}
        if session_id:
            payload["session_id"] = str(session_id)
        response = requests.post(f"{API_BASE_URL}/messages", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to send message: {e}")
        return None

def poll_messages():
    """Poll for new messages if a session is active."""
    if st.session_state.session_id:
        current_time = time.time()
        # Poll every 1 second
        if current_time - st.session_state.last_poll_time > 1:
            messages = fetch_messages(st.session_state.session_id)
            if len(messages) > st.session_state.last_message_count:
                st.session_state.messages = messages
                st.session_state.last_message_count = len(messages)
                st.session_state.last_poll_time = current_time
                st.rerun()

def handle_message_input():
    """Handle message input and initiate polling."""
    user_input = st.session_state[st.session_state.input_key]
    if user_input:
        # Optimistically add user message to UI
        st.session_state.messages.append({
            "session_id": st.session_state.session_id,
            "sender_type": "user",
            "content": user_input
        })
        # Update message count after adding user message
        st.session_state.last_message_count = len(st.session_state.messages)
        # Send message to backend
        send_message(user_input, st.session_state.session_id)
        # Reset input by changing the key
        st.session_state.input_key = f"user_input_{st.session_state.last_message_count}"
        # Update poll time to trigger new check
        st.session_state.last_poll_time = time.time() - 1  # Force immediate poll
        # Handle new session case
        if not st.session_state.session_id:
            st.session_state.sessions = fetch_sessions()
            if st.session_state.sessions:
                st.session_state.session_id = st.session_state.sessions[0]["session_id"]
                st.session_state.messages = fetch_messages(st.session_state.session_id)
                st.session_state.last_message_count = len(st.session_state.messages)
                st.session_state.last_poll_time = time.time() - 1

def display_message(content: str, is_user: bool):
    """Display a chat message with custom styling."""
    if is_user:
        st.markdown(
            f"""
            <div style="
                background-color: #0084ff;
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                margin: 5px 0px 5px 50px;
                max-width: 80%;
                margin-left: 20%;
                word-wrap: break-word;
            ">
                {content}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color: #f1f1f1;
                color: black;
                padding: 10px 15px;
                border-radius: 20px;
                margin: 5px 50px 5px 0px;
                max-width: 80%;
                word-wrap: break-word;
            ">
                {content}
            </div>
            """,
            unsafe_allow_html=True
        )

# Streamlit app layout
st.title("AI Chatbot")

# Sidebar for session management
st.sidebar.header("Sessions")
if st.sidebar.button("Create New Session"):
    new_session = create_session()
    if new_session:
        st.session_state.sessions = fetch_sessions()
        st.session_state.session_id = new_session["session_id"]
        st.session_state.messages = []
        st.session_state.last_poll_time = time.time() - 1
        st.session_state.input_key = "user_input_0"
        st.session_state.last_message_count = 0
        st.success("New session created!")

# Fetch and display sessions
st.session_state.sessions = fetch_sessions()
session_options = {f"{s['title'] or 'Untitled'} ({s['session_id'][:8]})": s["session_id"] for s in st.session_state.sessions}
selected_session = st.sidebar.selectbox("Select Session", options=[""] + list(session_options.keys()))

if selected_session:
    new_session_id = session_options[selected_session]
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        st.session_state.messages = fetch_messages(st.session_state.session_id)
        st.session_state.last_message_count = len(st.session_state.messages)
        st.session_state.last_poll_time = time.time() - 1
        st.session_state.input_key = "user_input_0"

# Poll for new messages
poll_messages()

# Chat interface
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        display_message(msg["content"], msg["sender_type"] == "user")

# Message input
st.text_input("Type your message...", key=st.session_state.input_key, on_change=handle_message_input)