from pydantic import BaseModel
from uuid import UUID
from typing import Optional

class MessageCreate(BaseModel):
    session_id: Optional[UUID] = None
    content: str

class MessageResponse(BaseModel):
    id: int
    session_id: UUID
    sender_type: str
    content: str
    created_at: str

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: UUID
    title: Optional[str]
    created_at: str
    updated_at: str