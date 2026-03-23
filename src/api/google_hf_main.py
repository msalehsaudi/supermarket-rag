"""FastAPI app using Google Gemini + Hugging Face APIs."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

# Import Google/Hugging Face chains
from src.chains.google_hf_chains import (
    google_classify_intent,
    google_meal_plan_chain,
    google_budget_optimizer_chain,
    google_product_search_chain
)

# Simple models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    conversation_id: str


class StatsResponse(BaseModel):
    total_products: int
    categories: int
    brands: int
    price_range: Dict[str, float]
    api_config: Dict[str, Any]


# Global state
app_state = {
    "start_time": time.time(),
    "active_conversations": {}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    print("🚀 Starting Google/Hugging Face API...")
    yield
    print("🛑 Google/Hugging Face API shutdown")


# Create app
app = FastAPI(
    title="Supermarket RAG API - Google + Hugging Face",
    description="AI-powered supermarket assistant using Google Gemini + Hugging Face",
    version="1.0.0-google-hf",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conversation_context(request: ChatRequest) -> Dict[str, Any]:
    """Get or create conversation context."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if conversation_id not in app_state["active_conversations"]:
        app_state["active_conversations"][conversation_id] = {
            "created_at": time.time(),
            "message_count": 0,
            "last_activity": time.time()
        }
    
    context = app_state["active_conversations"][conversation_id]
    context["message_count"] += 1
    context["last_activity"] = time.time()
    
    return {
        "conversation_id": conversation_id,
        "context": context
    }


@app.get("/")
async def root():
    """Root endpoint."""
    from src.config_google_hf import get_api_config
    api_config = get_api_config()
    
    return {
        "name": "Supermarket RAG API - Google + Hugging Face",
        "version": "1.0.0-google-hf",
        "description": "AI-powered supermarket assistant",
        "apis": {
            "llm": api_config["config"]["provider"],
            "embeddings": "Hugging Face"
        },
        "features": [
            "AI-powered meal planning",
            "Intelligent budget optimization", 
            "Advanced product search",
            "Nutrition advice",
            "Semantic search"
        ],
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "stats": "/stats"
        },
        "uptime": time.time() - app_state["start_time"]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "version": "1.0.0-google-hf",
        "uptime": time.time() - app_state["start_time"],
        "mode": "google-huggingface",
        "features": ["ai_llm", "embeddings", "semantic_search"]
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        from src.ingest.loader import load_and_clean
        from src.config_google_hf import get_api_config
        
        df = load_and_clean()
        api_config = get_api_config()
        
        return StatsResponse(
            total_products=len(df),
            categories=df['category'].nunique(),
            brands=df['brand'].nunique(),
            price_range={
                "min": float(df['price_eur'].min()),
                "max": float(df['price_eur'].max())
            },
            api_config=api_config["config"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using Google Gemini + Hugging Face."""
    conversation_context = get_conversation_context(request)
    
    try:
        # Classify intent
        intent_result = await google_classify_intent(request.message)
        intent = intent_result["intent"]
        constraints = intent_result["constraints"]
        
        print(f"🎯 Intent: {intent}, Constraints: {constraints}")
        
        # Route to appropriate chain
        if intent == "meal_plan":
            response_generator = google_meal_plan_chain(request.message, constraints)
        elif intent == "budget_basket":
            response_generator = google_budget_optimizer_chain(request.message, constraints)
        elif intent == "product_search":
            response_generator = google_product_search_chain(request.message, constraints)
        else:
            # General response
            response_generator = google_meal_plan_chain(request.message, constraints)
        
        # Stream response
        async def generate_response():
            full_response = ""
            async for chunk in response_generator:
                full_response += chunk
                yield chunk
            # Don't return, just yield the complete response at the end
            yield full_response
        
        # Return streaming response
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "X-Conversation-ID": conversation_context["conversation_id"],
                "X-Intent": intent,
                "X-Constraints": str(constraints)
            }
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/conversations")
async def list_conversations():
    """List active conversations."""
    return {
        "active_conversations": len(app_state["active_conversations"]),
        "conversations": [
            {
                "id": conv_id,
                "created_at": conv["created_at"],
                "message_count": conv["message_count"],
                "last_activity": conv["last_activity"]
            }
            for conv_id, conv in app_state["active_conversations"].items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Google/Hugging Face API...")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(
        "src.api.google_hf_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
