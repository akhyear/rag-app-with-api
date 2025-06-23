from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Literal

class MessageCreate(BaseModel):
    session_id: UUID
    content: str

class MessageResponse(BaseModel):
    message_id: UUID
    session_id: UUID
    sender_type: Literal["user", "ai"]
    content: str
    created_at: datetime

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime