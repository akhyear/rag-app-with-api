from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client
from .database import supabase
from .models import MessageCreate, MessageResponse, SessionResponse
from .rag import process_query
from .utils import generate_session_title
from uuid import UUID, uuid4
from typing import List, Optional
from datetime import datetime
import uvicorn  # Added for running the server
import os  # Added for environment variable access

app = FastAPI()
# security = HTTPBearer()

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     try:
#         # Verify Supabase JWT token
#         user = supabase.auth.get_user(credentials.credentials)
#         if not user:
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
#         return user.user.id
#     except Exception:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/sessions", response_model=SessionResponse)
async def create_session(user_id: str = "00000000-0000-0000-0000-000000000000"):  # Placeholder UUID
    # Create a new session
    session_id = str(uuid4())
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "title": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    response = supabase.table("chat_sessions").insert(session_data).execute()
    return response.data[0]

@app.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(user_id: str = "00000000-0000-0000-0000-000000000000"):  # Placeholder UUID
    response = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
    return response.data

@app.get("/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(session_id: UUID, user_id: str = "00000000-0000-0000-0000-000000000000"):  # Placeholder UUID
    # Ensure session belongs to user
    session = supabase.table("chat_sessions").select("*").eq("session_id", str(session_id)).eq("user_id", user_id).single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")
    
    response = supabase.table("chat_messages").select("*").eq("session_id", str(session_id)).order("created_at").execute()
    return response.data

@app.post("/messages", response_model=MessageResponse)
async def send_message(message: MessageCreate, user_id: str = "00000000-0000-0000-0000-000000000000"):  # Placeholder UUID
    session_id = message.session_id

    # If no session_id is provided, create a new session
    if not session_id:
        session_data = {
            "session_id": str(uuid4()),
            "user_id": user_id,
            "title": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        session = supabase.table("chat_sessions").insert(session_data).execute().data[0]
        session_id = session["session_id"]
    else:
        # Verify session exists and belongs to user
        session = supabase.table("chat_sessions").select("*").eq("session_id", str(session_id)).eq("user_id", user_id).single().execute()
        if not session.data:
            raise HTTPException(status_code=404, detail="Session not found or unauthorized")

    # Insert user message
    user_message = supabase.table("chat_messages").insert({
        "session_id": str(session_id),
        "sender_type": "user",
        "content": message.content,
        "created_at": datetime.utcnow().isoformat()
    }).execute().data[0]

    # Generate AI response
    ai_response_content = process_query(message.content)

    # Insert AI response
    ai_message = supabase.table("chat_messages").insert({
        "session_id": str(session_id),
        "sender_type": "ai",
        "content": ai_response_content,
        "created_at": datetime.utcnow().isoformat()
    }).execute().data[0]

    # Update session title if first message or title is None
    if not session.get("title") if isinstance(session, dict) else not session.data.get("title"):
        title = generate_session_title(message.content)
        supabase.table("chat_sessions").update({"title": title, "updated_at": datetime.utcnow().isoformat()}).eq("session_id", str(session_id)).execute()

    return ai_message

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
