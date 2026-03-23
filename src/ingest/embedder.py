"""Batch embedder with checkpointing for supermarket dataset."""

import json
import asyncio
from pathlib import Path
from typing import List, Set
import openai
from openai import AsyncOpenAI
from langchain_core.documents import Document

from src.config import (
    OPENAI_API_KEY, 
    EMBEDDING_MODEL, 
    EMBEDDING_BATCH_SIZE,
    CHECKPOINT_PATH,
    COLLECTION_NAME
)
from src.vectorstore.chroma_store import get_collection
from .loader import load_and_clean
from .doc_builder import build_documents


def load_checkpoint() -> Set[int]:
    """
    Load checkpoint of already embedded product IDs.
    
    Returns:
        Set of product IDs that have been embedded
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
            return set(data.get('embedded_ids', []))
    return set()


def save_checkpoint(embedded_ids: Set[int]) -> None:
    """
    Save checkpoint of embedded product IDs.
    
    Args:
        embedded_ids: Set of product IDs that have been embedded
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({
            'embedded_ids': list(embedded_ids),
            'last_updated': str(Path().cwd()),
            'model': EMBEDDING_MODEL,
            'collection': COLLECTION_NAME
        }, f, indent=2)


async def embed_batch(texts: List[str], client: AsyncOpenAI) -> List[List[float]]:
    """
    Embed a batch of texts using OpenAI API.
    
    Args:
        texts: List of texts to embed
        client: OpenAI async client
        
    Returns:
        List of embedding vectors
    """
    try:
        response = await client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [r.embedding for r in response.data]
    except Exception as e:
        print(f"Error embedding batch: {e}")
        raise


async def embed_and_upsert(documents: List[Document]) -> None:
    """
    Embed documents and upsert to ChromaDB with checkpointing.
    
    Args:
        documents: List of Document objects to embed
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for embeddings")
    
    # Load checkpoint and filter already processed documents
    checkpoint = load_checkpoint()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    collection = get_collection()
    
    pending_docs = [
        doc for doc in documents 
        if doc.metadata.get('product_id') not in checkpoint
    ]
    
    if not pending_docs:
        print("All documents already embedded. Nothing to do.")
        return
    
    print(f"Embedding {len(pending_docs)} new documents (skipping {len(documents) - len(pending_docs)} already embedded)")
    
    # Process in batches
    batches = [
        pending_docs[i:i + EMBEDDING_BATCH_SIZE] 
        for i in range(0, len(pending_docs), EMBEDDING_BATCH_SIZE)
    ]
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} documents)")
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [str(doc.metadata['product_id']) for doc in batch]
        
        # Embed the batch
        embeddings = await embed_batch(texts, client)
        
        # Upsert to ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        # Update checkpoint
        for doc in batch:
            checkpoint.add(doc.metadata['product_id'])
        
        save_checkpoint(checkpoint)
        print(f"Batch {i + 1} completed. Total embedded: {len(checkpoint)}")
    
    print(f"Embedding completed! Total products in vector store: {len(checkpoint)}")


async def run_ingestion_pipeline() -> None:
    """
    Run the complete ingestion pipeline.
    """
    print("Starting ingestion pipeline...")
    
    # Load and clean data
    df = load_and_clean()
    print(f"Loaded {len(df)} products from CSV")
    
    # Build documents
    documents = build_documents(df)
    print(f"Built {len(documents)} documents")
    
    # Embed and upsert
    await embed_and_upsert(documents)
    
    print("Ingestion pipeline completed successfully!")


def get_ingestion_stats() -> dict:
    """
    Get statistics about the ingestion process.
    
    Returns:
        Dictionary with ingestion statistics
    """
    checkpoint = load_checkpoint()
    
    stats = {
        "embedded_products": len(checkpoint),
        "checkpoint_file": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "embedding_model": EMBEDDING_MODEL,
        "batch_size": EMBEDDING_BATCH_SIZE
    }
    
    # Get vector store stats if available
    try:
        collection = get_collection()
        stats["vector_store_count"] = collection.count()
        stats["collection_name"] = COLLECTION_NAME
    except Exception as e:
        stats["vector_store_error"] = str(e)
    
    return stats


if __name__ == "__main__":
    # Run the ingestion pipeline
    asyncio.run(run_ingestion_pipeline())
