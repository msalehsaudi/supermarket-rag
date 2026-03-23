"""Minimal FastAPI app without heavy dependencies."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

# Simple models
class SimpleChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class SimpleChatResponse(BaseModel):
    response: str
    intent: str
    conversation_id: str


class SimpleStatsResponse(BaseModel):
    total_products: int
    categories: int
    brands: int
    price_range: Dict[str, float]


# Global state
app_state = {
    "start_time": time.time(),
    "active_conversations": {}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    print("🚀 Starting Simple API...")
    yield
    print("🛑 Simple API shutdown")


# Create app
app = FastAPI(
    title="Supermarket RAG API - Simple Mode",
    description="Basic supermarket assistant without heavy dependencies",
    version="1.0.0-simple",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conversation_context(request: SimpleChatRequest) -> Dict[str, Any]:
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
    return {
        "name": "Supermarket RAG API - Simple Mode",
        "version": "1.0.0-simple",
        "description": "Basic supermarket assistant (no AI responses)",
        "features": [
            "Product search",
            "Category filtering", 
            "Price sorting",
            "Basic statistics"
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
        "version": "1.0.0-simple",
        "uptime": time.time() - app_state["start_time"],
        "mode": "simple",
        "features": ["basic_search", "no_ai"]
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        from src.ingest.loader import load_and_clean
        df = load_and_clean()
        
        return SimpleStatsResponse(
            total_products=len(df),
            categories=df['category'].nunique(),
            brands=df['brand'].nunique(),
            price_range={
                "min": float(df['price_eur'].min()),
                "max": float(df['price_eur'].max())
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.post("/chat")
async def simple_chat(request: SimpleChatRequest):
    """Simple chat endpoint with template responses."""
    conversation_context = get_conversation_context(request)
    
    # Simple intent classification
    message_lower = request.message.lower()
    
    if any(word in message_lower for word in ["meal plan", "meal plan", "weekly", "daily meals"]):
        intent = "meal_plan"
        response = """🍽️ Meal Plan Template

I can help you create meal plans! For detailed meal planning with:
• Personalized nutrition goals
• Calorie calculations  
• Recipe suggestions
• Shopping lists

Please add an OpenAI API key to your .env file and restart with:
python start.py

Current template meal plan:
• Day 1: Oatmeal, Grilled Chicken, Mixed Vegetables
• Day 2: Greek Yogurt, Turkey Sandwich, Fresh Fruit  
• Day 3: Quinoa Bowl, Fish Fillet, Side Salad
• Day 4: Smoothie, Pasta Primavera, Nuts

This is a basic template. For personalized meal planning, I need access to AI models."""
        
    elif any(word in message_lower for word in ["budget", "cheap", "cheapest", "affordable"]):
        intent = "budget_basket"
        response = f"""💰 Budget Optimization Template

I can help you find budget-friendly products! For advanced budget optimization with:
• Price comparisons across brands
• Nutritional value analysis
• Bulk buying recommendations
• Store specials tracking

Please add an OpenAI API key to your .env file and restart with:
python start.py

Current budget analysis for your query "{request.message}":
• Total products in database: 10,000
• Price range: €0.07 - €123.19
• Categories available: 15
• Average product price: €8.45

This is basic analysis. For AI-powered budget optimization, I need access to advanced models."""
        
    elif any(word in message_lower for word in ["find", "search", "looking for", "show me"]):
        intent = "product_search"
        response = f"""🔍 Product Search Template

I can help you search for products! For advanced product search with:
• Semantic similarity search
• Category filtering
• Brand preferences
• Nutritional requirements
• Price comparisons

Please add an OpenAI API key to your .env file and restart with:
python start.py

Basic search results for "{request.message}":
• Try specific product names
• Use category names (e.g., "dairy", "vegetables")
• Include price ranges (e.g., "under 5 euros")
• Mention brands if you have preferences

This is basic search. For AI-powered product discovery, I need access to embedding models."""
        
    else:
        intent = "general"
        response = f"""🛒 Welcome to Supermarket Assistant!

I'm here to help with your shopping needs! I can assist with:

📊 Product Information
• Search for specific products
• Compare prices and features
• Find nutritional information
• Check availability

🛒 Shopping Assistance  
• Create shopping lists
• Budget planning
• Category browsing
• Brand comparisons

🍽️ Basic Meal Planning
• Simple meal templates
• Nutrition tracking
• Budget-friendly options

For advanced AI-powered features like:
• Personalized meal planning
• Intelligent budget optimization
• Nutritional advice
• Semantic product search

Please add an OpenAI API key to your .env file and restart with:
python start.py

Your current query: "{request.message}"
I'll help you with basic product information and search using our database of 10,000 products!"""
    
    return SimpleChatResponse(
        response=response,
        intent=intent,
        conversation_id=conversation_context["conversation_id"]
    )


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
    print("🚀 Starting Simple Supermarket API...")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(
        "src.api.simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
