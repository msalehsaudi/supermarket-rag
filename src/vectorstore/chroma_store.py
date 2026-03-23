"""ChromaDB vector store for supermarket products."""

import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from typing import Dict, Any, List, Optional

from src.config import (
    CHROMA_DB_PATH, 
    OPENAI_API_KEY, 
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    METADATA_FIELDS
)


def get_client():
    """
    Get ChromaDB persistent client.
    
    Returns:
        ChromaDB client
    """
    # Ensure the directory exists
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_embedding_function():
    """
    Get OpenAI embedding function.
    
    Returns:
        OpenAI embedding function
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for embeddings")
    
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )


def get_collection():
    """
    Get or create the supermarket products collection.
    
    Returns:
        ChromaDB collection
    """
    client = get_client()
    embedding_fn = get_embedding_function()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def build_metadata_filter(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a ChromaDB $where filter from extracted constraints.
    
    Args:
        constraints: Dictionary of query constraints
        
    Returns:
        ChromaDB filter dictionary
        
    Examples:
        Input: {"max_price": 5.0, "min_protein": 15.0, "food_only": True, "in_stock": True}
        Output: {"$and": [
            {"is_food": {"$eq": True}},
            {"price_eur": {"$lte": 5.0}},
            {"protein_g_per_100g": {"$gte": 15.0}},
            {"in_stock": {"$eq": True}}
        ]}
    """
    filters = []
    
    # Food filter
    if constraints.get("food_only"):
        filters.append({"is_food": {"$eq": True}})
    
    # Stock filter
    if constraints.get("in_stock"):
        filters.append({"in_stock": {"$eq": True}})
    
    # Price filters
    if "max_price" in constraints:
        filters.append({"price_eur": {"$lte": constraints["max_price"]}})
    if "min_price" in constraints:
        filters.append({"price_eur": {"$gte": constraints["min_price"]}})
    
    # Nutrition filters
    if "min_protein" in constraints:
        filters.append({"protein_g_per_100g": {"$gte": constraints["min_protein"]}})
    if "max_calories" in constraints:
        filters.append({"calories_per_100g": {"$lte": constraints["max_calories"]}})
    if "max_sugar" in constraints:
        filters.append({"sugar_g_per_100g": {"$lte": constraints["max_sugar"]}})
    if "min_fiber" in constraints:
        filters.append({"fiber_g_per_100g": {"$gte": constraints["min_fiber"]}})
    
    # Category filter
    if "category" in constraints:
        filters.append({"category": {"$eq": constraints["category"]}})
    
    # Brand filter
    if "brand" in constraints:
        filters.append({"brand": {"$eq": constraints["brand"]}})
    
    # Origin filter
    if "origin" in constraints:
        filters.append({"origin": {"$eq": constraints["origin"]}})
    
    # Rating filter
    if "min_rating" in constraints:
        filters.append({"rating": {"$gte": constraints["min_rating"]}})
    
    # Weight filter
    if "max_weight" in constraints:
        filters.append({"weight_kg": {"$lte": constraints["max_weight"]}})
    if "min_weight" in constraints:
        filters.append({"weight_kg": {"$gte": constraints["min_weight"]}})
    
    # Combine filters
    if len(filters) == 0:
        return {}
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


def query_collection(
    query_text: str,
    where_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 20,
    include: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Query the ChromaDB collection.
    
    Args:
        query_text: Query text to search for
        where_filter: Metadata filter dictionary
        n_results: Number of results to return
        include: List of fields to include in response
        
    Returns:
        Query results from ChromaDB
    """
    collection = get_collection()
    
    if include is None:
        include = ["metadatas", "documents", "distances"]
    
    results = collection.query(
        query_texts=[query_text],
        where=where_filter,
        n_results=n_results,
        include=include
    )
    
    return results


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the collection.
    
    Returns:
        Dictionary with collection statistics
    """
    collection = get_collection()
    
    try:
        count = collection.count()
        
        # Get sample to infer metadata fields
        sample_results = collection.get(limit=1, include=["metadatas"])
        
        stats = {
            "name": COLLECTION_NAME,
            "count": count,
            "embedding_model": EMBEDDING_MODEL,
            "db_path": str(CHROMA_DB_PATH),
            "metadata_fields": list(METADATA_FIELDS.keys())
        }
        
        if sample_results["metadatas"]:
            stats["sample_metadata"] = sample_results["metadatas"][0]
        
        return stats
        
    except Exception as e:
        return {
            "error": str(e),
            "name": COLLECTION_NAME,
            "db_path": str(CHROMA_DB_PATH)
        }


def reset_collection() -> None:
    """
    Delete and recreate the collection.
    WARNING: This will delete all embedded data!
    """
    client = get_client()
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted collection: {COLLECTION_NAME}")
    except Exception:
        print(f"Collection {COLLECTION_NAME} didn't exist or couldn't be deleted")
    
    # Recreate empty collection
    get_collection()
    print(f"Recreated empty collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    # Test the vector store
    stats = get_collection_stats()
    print("Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
