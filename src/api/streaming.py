"""Streaming utilities for Server-Sent Events (SSE)."""

import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi.responses import StreamingResponse
from .models import StreamChunk


async def stream_response(
    content_generator: AsyncGenerator[str, None],
    metadata: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None
) -> StreamingResponse:
    """
    Create a streaming response from a content generator.
    
    Args:
        content_generator: Async generator yielding content chunks
        metadata: Optional metadata to include
        intent: Optional intent to include
        
    Returns:
        FastAPI StreamingResponse for SSE
    """
    
    async def generate_sse():
        """Generate Server-Sent Events."""
        try:
            # Send initial metadata if provided
            if metadata or intent:
                chunk_data = {"type": "metadata"}
                if metadata:
                    chunk_data["metadata"] = metadata
                if intent:
                    chunk_data["metadata"] = chunk_data.get("metadata", {})
                    chunk_data["metadata"]["intent"] = intent
                
                chunk = StreamChunk(**chunk_data)
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Stream content chunks
            async for content_chunk in content_generator:
                chunk = StreamChunk(type="content", content=content_chunk)
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
            # Send completion signal
            chunk = StreamChunk(type="done", done=True)
            yield f"data: {chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            # Send error chunk
            error_chunk = StreamChunk(
                type="error",
                error=str(e)
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


async def stream_json_response(
    data_generator: AsyncGenerator[Dict[str, Any], None]
) -> StreamingResponse:
    """
    Create a streaming JSON response.
    
    Args:
        data_generator: Async generator yielding data dictionaries
        
    Returns:
        FastAPI StreamingResponse for SSE
    """
    
    async def generate_sse():
        """Generate Server-Sent Events with JSON data."""
        try:
            async for data_chunk in data_generator:
                chunk = StreamChunk(type="content", content=json.dumps(data_chunk))
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)
            
            # Send completion signal
            chunk = StreamChunk(type="done", done=True)
            yield f"data: {chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            error_chunk = StreamChunk(type="error", error=str(e))
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


async def stream_error_response(error_message: str) -> StreamingResponse:
    """
    Create a streaming error response.
    
    Args:
        error_message: Error message to send
        
    Returns:
        FastAPI StreamingResponse for SSE
    """
    
    async def generate_sse():
        """Generate error SSE."""
        chunk = StreamChunk(type="error", error=error_message)
        yield f"data: {chunk.model_dump_json()}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


class StreamingContext:
    """
    Context manager for streaming operations with timing and metadata.
    """
    
    def __init__(self, intent: Optional[str] = None):
        """
        Initialize streaming context.
        
        Args:
            intent: Detected intent for the request
        """
        self.intent = intent
        self.start_time = None
        self.metadata = {
            "intent": intent,
            "start_time": None,
            "chunks_sent": 0,
            "total_chars": 0
        }
    
    async def __aenter__(self):
        """Enter context."""
        import time
        self.start_time = time.time()
        self.metadata["start_time"] = self.start_time
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        import time
        if self.start_time:
            self.metadata["response_time"] = time.time() - self.start_time
    
    async def wrap_generator(self, generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Wrap a content generator to collect metadata.
        
        Args:
            generator: Original content generator
            
        Yields:
            Content chunks with metadata tracking
        """
        async for chunk in generator:
            self.metadata["chunks_sent"] += 1
            self.metadata["total_chars"] += len(chunk)
            yield chunk
    
    def get_final_metadata(self) -> Dict[str, Any]:
        """
        Get final metadata after streaming.
        
        Returns:
            Complete metadata dictionary
        """
        return self.metadata.copy()


# Utility functions for different response types
async def stream_meal_plan_response(
    generator: AsyncGenerator[str, None],
    constraints: Dict[str, Any]
) -> StreamingResponse:
    """
    Stream a meal plan response with meal-specific metadata.
    
    Args:
        generator: Content generator
        constraints: User constraints
        
    Returns:
        StreamingResponse with meal plan metadata
    """
    metadata = {
        "type": "meal_plan",
        "constraints": constraints,
        "days": constraints.get("days", 7),
        "budget": constraints.get("budget", 0)
    }
    
    return await stream_response(generator, metadata, "meal_plan")


async def stream_budget_response(
    generator: AsyncGenerator[str, None],
    constraints: Dict[str, Any]
) -> StreamingResponse:
    """
    Stream a budget optimization response with budget-specific metadata.
    
    Args:
        generator: Content generator
        constraints: User constraints
        
    Returns:
        StreamingResponse with budget metadata
    """
    metadata = {
        "type": "budget_optimization",
        "constraints": constraints,
        "budget": constraints.get("budget", 0),
        "goal": constraints.get("goal", "general")
    }
    
    return await stream_response(generator, metadata, "budget_basket")


async def stream_nutrition_response(
    generator: AsyncGenerator[str, None],
    constraints: Dict[str, Any]
) -> StreamingResponse:
    """
    Stream a nutrition response with nutrition-specific metadata.
    
    Args:
        generator: Content generator
        constraints: User constraints
        
    Returns:
        StreamingResponse with nutrition metadata
    """
    metadata = {
        "type": "nutrition_advice",
        "constraints": constraints,
        "focus": constraints.get("nutrition_focus", "general")
    }
    
    return await stream_response(generator, metadata, "nutrition_query")


async def stream_product_search_response(
    generator: AsyncGenerator[str, None],
    constraints: Dict[str, Any]
) -> StreamingResponse:
    """
    Stream a product search response with search-specific metadata.
    
    Args:
        generator: Content generator
        constraints: User constraints
        
    Returns:
        StreamingResponse with search metadata
    """
    metadata = {
        "type": "product_search",
        "constraints": constraints,
        "category": constraints.get("category", "all"),
        "brand": constraints.get("brand", "all")
    }
    
    return await stream_response(generator, metadata, "product_search")


async def stream_general_response(
    generator: AsyncGenerator[str, None],
    constraints: Dict[str, Any]
) -> StreamingResponse:
    """
    Stream a general response with general metadata.
    
    Args:
        generator: Content generator
        constraints: User constraints
        
    Returns:
        StreamingResponse with general metadata
    """
    metadata = {
        "type": "general_query",
        "constraints": constraints
    }
    
    return await stream_response(generator, metadata, "general")
