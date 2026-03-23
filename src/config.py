"""Configuration for Supermarket RAG System using Google Gemini + Hugging Face."""

import os
from pathlib import Path
from typing import Final

# API Keys
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

# Paths
BASE_DIR: Final[Path] = Path(__file__).parent.parent
DATA_PATH: Final[Path] = BASE_DIR / "data" / "supermarket_dataset.csv"
CHROMA_DB_PATH: Final[Path] = BASE_DIR / "chroma_db"
CHECKPOINT_PATH: Final[Path] = CHROMA_DB_PATH / ".ingest_checkpoint.json"

# Embedding settings (using Hugging Face)
HF_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_DIM: Final[int] = 384
EMBEDDING_BATCH_SIZE: Final[int] = 50

# LLM settings (using Google Gemini)
GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
LLM_TEMPERATURE: Final[float] = 0.1
LLM_MAX_TOKENS: Final[int] = 4000

# Retrieval settings
RETRIEVAL_TOP_K: Final[int] = 20
RERANK_TOP_K: Final[int] = 5

# API settings
API_PORT: int = int(os.getenv("API_PORT", "8002"))
API_HOST: str = os.getenv("API_HOST", "localhost")

# Collection name
COLLECTION_NAME: Final[str] = "supermarket_products"

# Metadata schema
METADATA_FIELDS = {
    "product_id": int,
    "sku": str,
    "name": str,
    "brand": str,
    "category": str,
    "origin": str,
    "weight_kg": float,
    "price_eur": float,
    "in_stock": bool,
    "rating": float,
    "is_food": bool,
    "calories_per_100g": float,
    "protein_g_per_100g": float,
    "fat_g_per_100g": float,
    "sugar_g_per_100g": float,
    "carbs_g_per_100g": float,
    "fiber_g_per_100g": float,
    "sodium_mg_per_100g": float,
}

def get_api_config():
    """Get current API configuration."""
    if GOOGLE_API_KEY:
        return {"type": "google", "config": {"model": GEMINI_MODEL, "provider": "Google Gemini"}}
    elif HUGGINGFACE_API_KEY:
        return {"type": "huggingface", "config": {"model": HF_EMBEDDING_MODEL, "provider": "Hugging Face"}}
    else:
        return {"type": "none", "config": {"error": "No API keys found"}}

def get_embedding_function():
    """Get embedding function using Hugging Face."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )
    except ImportError:
        # Fallback to local sentence transformers
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(HF_EMBEDDING_MODEL)

def get_llm_config():
    """Get LLM configuration using Google Gemini."""
    return {
        "provider": "google",
        "model": GEMINI_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "api_key": GOOGLE_API_KEY
    }

# Validate API keys
if not GOOGLE_API_KEY:
    print("⚠️  GOOGLE_API_KEY not set - LLM features will not work")
if not HUGGINGFACE_API_KEY:
    print("⚠️  HUGGINGFACE_API_KEY not set - using local embeddings")

print(f"🔧 Using APIs:")
print(f"  🤖 LLM: Google Gemini ({GEMINI_MODEL})")
print(f"  📊 Embeddings: Hugging Face ({HF_EMBEDDING_MODEL})")
print(f"  📏 Embedding Dim: {HF_EMBEDDING_DIM}")
