from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client
from app.database import supabase
from app.models import MessageCreate, MessageResponse, SessionResponse
from app.langchain_rag import get_rag_service, RAGService
from uuid import UUID, uuid4
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="LangChain RAG Chat API",
    description="FastAPI application with LangChain RAG using Groq and Pinecone",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service on startup
rag_service: Optional[RAGService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG service and index documents if needed"""
    global rag_service
    try:
        logger.info("Initializing RAG service...")
        rag_service = get_rag_service()
        
        # Index documents in background if needed
        if not rag_service.is_index_populated():
            logger.info("Index is empty, starting document indexing...")
            rag_service.index_documents()
        
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Don't fail startup, but log the error
        rag_service = None

def get_rag_service_instance() -> RAGService:
    """Get RAG service instance with error handling"""
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not available. Please check server logs."
        )
    return rag_service

@app.post("/sessions", response_model=SessionResponse)
async def create_session(user_id: str = "00000000-0000-0000-0000-000000000000"):
    """Create a new chat session"""
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
async def get_sessions(user_id: str = "00000000-0000-0000-0000-000000000000"):
    """Get all sessions for a user"""
    response = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
    return response.data

@app.get("/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(session_id: UUID, user_id: str = "00000000-0000-0000-0000-000000000000"):
    """Get all messages for a session"""
    # Ensure session belongs to user
    session = supabase.table("chat_sessions").select("*").eq("session_id", str(session_id)).eq("user_id", user_id).single().execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")
    
    response = supabase.table("chat_messages").select("*").eq("session_id", str(session_id)).order("created_at").execute()
    return response.data

@app.post("/messages", response_model=MessageResponse)
async def send_message(message: MessageCreate, user_id: str = "00000000-0000-0000-0000-000000000000"):
    """Send a message and get AI response using RAG"""
    session_id = message.session_id
    rag = get_rag_service_instance()

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

    try:
        # Generate AI response using RAG
        rag_result = rag.retrieve_answers(message.content, include_sources=True)
        ai_response_content = rag_result.get("answer", "I couldn't find a relevant answer.")
        
        '''# Add sources information if available
        if "sources" in rag_result and rag_result["sources"]:
            sources_text = "\n\nSources:\n"
            for i, source in enumerate(rag_result["sources"][:2], 1):  # Limit to 2 sources
                sources_text += f"{i}. {source['source']} (Page {source['page']})\n"
            ai_response_content += sources_text'''
            
    except Exception as e:
        logger.error(f"RAG processing failed: {e}")
        ai_response_content = "I'm sorry, I encountered an error while processing your question. Please try again."

    # Insert AI response
    ai_message = supabase.table("chat_messages").insert({
        "session_id": str(session_id),
        "sender_type": "ai",
        "content": ai_response_content,
        "created_at": datetime.utcnow().isoformat()
    }).execute().data[0]

    # Update session title if first message or title is None
    session_data = session.data if hasattr(session, 'data') else session
    if not session_data.get("title"):
        # Generate title from first message
        title_words = message.content.split()[:5]  # First 5 words
        title = " ".join(title_words)
        if len(message.content.split()) > 5:
            title += "..."
        
        supabase.table("chat_sessions").update({
            "title": title[:50],  # Limit title length
            "updated_at": datetime.utcnow().isoformat()
        }).eq("session_id", str(session_id)).execute()

    return ai_message

@app.post("/rag/query")
async def rag_query(query: str) -> Dict[str, Any]:
    """Direct RAG query endpoint"""
    rag = get_rag_service_instance()
    
    try:
        result = rag.retrieve_answers(query, include_sources=True)
        return {
            "success": True,
            "query": query,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/rag/similarity-search")
async def similarity_search(query: str, k: int = 3) -> Dict[str, Any]:
    """Similarity search endpoint"""
    rag = get_rag_service_instance()
    
    try:
        results = rag.similarity_search(query, k=k)
        return {
            "success": True,
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.post("/rag/reindex")
async def reindex_documents(background_tasks: BackgroundTasks):
    """Reindex documents (admin endpoint)"""
    rag = get_rag_service_instance()
    
    def reindex_task():
        try:
            rag.index_documents(force_reindex=True)
            logger.info("Document reindexing completed")
        except Exception as e:
            logger.error(f"Document reindexing failed: {e}")
    
    background_tasks.add_task(reindex_task)
    return {"message": "Document reindexing started in background"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "unknown",
            "rag": "unknown"
        }
    }
    
    # Check database connection
    try:
        supabase.table("chat_sessions").select("session_id").limit(1).execute()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check RAG service
    if rag_service:
        try:
            rag_health = rag_service.health_check()
            health_status["services"]["rag"] = rag_health["status"]
            health_status["rag_details"] = rag_health
        except Exception as e:
            health_status["services"]["rag"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["services"]["rag"] = "not_initialized"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangChain RAG Chat API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/rag/stats")
async def rag_stats():
    """Get RAG service statistics"""
    rag = get_rag_service_instance()
    
    try:
        stats = rag.index.describe_index_stats()
        return {
            "index_name": rag.index_name,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )