from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client
from .database import supabase
from .models import MessageCreate, MessageResponse, SessionResponse
from .rag import process_query
from .utils import generate_session_title
from uuid import UUID
from typing import List

app = FastAPI()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify Supabase JWT token
        user = supabase.auth.get_user(credentials.credentials)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user.user.id
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(user_id: str = Depends(get_current_user)):
    response = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
    return response.data

@app.get("/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(session_id: UUID, user_id: str = Depends(get_current_user)):
    # Ensure session belongs to user
    session = supabase.table("chat_sessions").select("*").eq("session_id", str(session_id)).eq("user_id", user_id).single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")
    
    response = supabase.table("chat_messages").select("*").eq("session_id", str(session_id)).order("created_at").execute()
    return response.data

@app.post("/messages", response_model=MessageResponse)
async def send_message(message: MessageCreate, user_id: str = Depends(get_current_user)):
    # Verify session exists and belongs to user
    session = supabase.table("chat_sessions").select("*").eq("session_id", str(message.session_id)).eq("user_id", user_id).single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")

    # Insert user message
    user_message = supabase.table("chat_messages").insert({
        "session_id": str(message.session_id),
        "sender_type": "user",
        "content": message.content
    }).execute().data[0]

    # Generate AI response
    ai_response_content = process_query(message.content)

    # Insert AI response
    ai_message = supabase.table("chat_messages").insert({
        "session_id": str(message.session_id),
        "sender_type": "ai",
        "content": ai_response_content
    }).execute().data[0]

    # Update session title if first message
    if not session.data.get("title"):
        title = generate_session_title(message.content)
        supabase.table("chat_sessions").update({"title": title, "updated_at": "now()"}).eq("session_id", str(message.session_id)).execute()

    return ai_message
# python -m uvicorn main:app --reload