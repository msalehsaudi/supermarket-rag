"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    """Possible intent types."""
    MEAL_PLAN = "meal_plan"
    BUDGET_BASKET = "budget_basket"
    NUTRITION_QUERY = "nutrition_query"
    PRODUCT_SEARCH = "product_search"
    GENERAL = "general"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message", min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    history: Optional[List[Dict[str, str]]] = Field(
        default=[], 
        description="Conversation history: [{'role': 'user/assistant', 'content': str}]"
    )
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant response")
    intent: IntentType = Field(..., description="Detected intent")
    sources: List[Dict[str, Any]] = Field(default=[], description="Products/sources used in response")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata like cost, macros, etc.")
    conversation_id: str = Field(..., description="Conversation ID")
    response_time: Optional[float] = Field(None, description="Response time in seconds")


class IngestRequest(BaseModel):
    """Request model for data ingestion."""
    force_reingest: bool = Field(False, description="Force re-ingestion of all data")
    batch_size: Optional[int] = Field(None, description="Override batch size for embedding")


class IngestResponse(BaseModel):
    """Response model for data ingestion."""
    status: str = Field(..., description="Status: started, completed, failed")
    message: str = Field(..., description="Status message")
    total_products: Optional[int] = Field(None, description="Total products to process")
    processed_products: Optional[int] = Field(None, description="Number of processed products")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    estimated_time: Optional[float] = Field(None, description="Estimated completion time in minutes")


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    total_products: int = Field(..., description="Total products in vector store")
    food_products: int = Field(..., description="Number of food products")
    in_stock_products: int = Field(..., description="Number of in-stock products")
    categories: int = Field(..., description="Number of unique categories")
    brands: int = Field(..., description="Number of unique brands")
    embedding_model: str = Field(..., description="Embedding model used")
    chroma_path: str = Field(..., description="Path to ChromaDB")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class ProductInfo(BaseModel):
    """Product information model."""
    product_id: int
    sku: str
    name: str
    brand: str
    category: str
    origin: str
    weight_kg: float
    price_eur: float
    in_stock: bool
    rating: float
    is_food: bool
    calories_per_100g: Optional[float] = None
    protein_g_per_100g: Optional[float] = None
    fat_g_per_100g: Optional[float] = None
    sugar_g_per_100g: Optional[float] = None
    carbs_g_per_100g: Optional[float] = None
    fiber_g_per_100g: Optional[float] = None
    sodium_mg_per_100g: Optional[float] = None


class PriceUpdateRequest(BaseModel):
    """Request model for price updates."""
    product_ids: List[int] = Field(..., description="List of product IDs to update")
    new_prices: List[float] = Field(..., description="New prices for each product")
    reason: Optional[str] = Field(None, description="Reason for price update")


class PriceUpdateResponse(BaseModel):
    """Response model for price updates."""
    updated_products: int = Field(..., description="Number of successfully updated products")
    failed_updates: List[Dict[str, Any]] = Field(default=[], description="Failed updates with reasons")
    old_prices: List[Dict[str, Any]] = Field(default=[], description="Previous prices")
    timestamp: str = Field(..., description="Update timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status: healthy, unhealthy")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Dependency statuses")
    timestamp: str = Field(..., description="Check timestamp")


# Streaming response models (for SSE)
class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    type: str = Field(..., description="Chunk type: content, metadata, error, done")
    content: Optional[str] = Field(None, description="Content chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata chunk")
    error: Optional[str] = Field(None, description="Error message")
    done: Optional[bool] = Field(None, description="Stream completion flag")


# Request validation helpers
def validate_chat_request(request: ChatRequest) -> None:
    """Validate chat request."""
    if not request.message.strip():
        raise ValueError("Message cannot be empty")
    
    if len(request.message) > 1000:
        raise ValueError("Message too long (max 1000 characters)")


def validate_price_update_request(request: PriceUpdateRequest) -> None:
    """Validate price update request."""
    if len(request.product_ids) != len(request.new_prices):
        raise ValueError("Product IDs and prices must have the same length")
    
    for price in request.new_prices:
        if price < 0:
            raise ValueError("Prices cannot be negative")
    
    if len(request.product_ids) > 100:
        raise ValueError("Cannot update more than 100 products at once")
