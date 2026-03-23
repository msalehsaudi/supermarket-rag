"""Free embedder using Hugging Face or local models."""

import json
import asyncio
from pathlib import Path
from typing import List, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from src.config_free import (
    FREE_EMBEDDING_MODEL, 
    FREE_EMBEDDING_DIM, 
    EMBEDDING_BATCH_SIZE,
    CHECKPOINT_PATH,
    COLLECTION_NAME
)
from src.vectorstore.chroma_store import get_collection
from src.ingest.doc_builder import build_documents


class FreeEmbedder:
    """
    Free embedder using sentence-transformers (local) or Hugging Face API.
    """
    
    def __init__(self, use_huggingface: bool = False):
        """
        Initialize free embedder.
        
        Args:
            use_huggingface: Whether to use Hugging Face API (requires API key)
        """
        self.use_huggingface = use_huggingface
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            print(f"🔄 Loading embedding model: {FREE_EMBEDDING_MODEL}")
            self.model = SentenceTransformer(FREE_EMBEDDING_MODEL)
            print(f"✅ Model loaded successfully")
            print(f"📏 Embedding dimension: {FREE_EMBEDDING_DIM}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using local sentence transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            self._load_model()
        
        try:
            # Process in smaller batches to avoid memory issues
            embeddings = []
            batch_size = min(EMBEDDING_BATCH_SIZE, len(texts))
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.extend(batch_embeddings.tolist())
                
                if i + batch_size < len(texts):
                    print(f"📊 Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error embedding batch: {e}")
            raise
    
    async def embed_and_upsert(self, documents: List[Document]) -> None:
        """
        Embed documents and upsert to ChromaDB with checkpointing.
        
        Args:
            documents: List of Document objects to embed
        """
        if not documents:
            print("No documents to embed")
            return
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        
        # Filter already processed documents
        pending_docs = [
            doc for doc in documents 
            if doc.metadata.get('product_id') not in checkpoint
        ]
        
        if not pending_docs:
            print("All documents already embedded. Nothing to do.")
            return
        
        print(f"🔄 Embedding {len(pending_docs)} new documents (skipping {len(documents) - len(pending_docs)} already embedded)")
        
        # Extract texts
        texts = [doc.page_content for doc in pending_docs]
        
        # Embed in batches
        all_embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_docs = pending_docs[i:i + batch_size]
            
            print(f"📦 Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} ({len(batch_texts)} documents)")
            
            # Embed the batch
            batch_embeddings = self.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Prepare for upsert
            metadatas = [doc.metadata for doc in batch_docs]
            ids = [str(doc.metadata['product_id']) for doc in batch_docs]
            
            # Upsert to ChromaDB
            collection = get_collection()
            collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=metadatas
            )
            
            # Update checkpoint
            for doc in batch_docs:
                checkpoint.add(doc.metadata['product_id'])
            
            self.save_checkpoint(checkpoint)
            print(f"✅ Batch completed. Total embedded: {len(checkpoint)}")
        
        print(f"🎉 Embedding completed! Total products in vector store: {len(checkpoint)}")
    
    def load_checkpoint(self) -> Set[int]:
        """Load checkpoint of already embedded product IDs."""
        if CHECKPOINT_PATH.exists():
            with open(CHECKPOINT_PATH, 'r') as f:
                data = json.load(f)
                return set(data.get('embedded_ids', []))
        return set()
    
    def save_checkpoint(self, embedded_ids: Set[int]) -> None:
        """Save checkpoint of embedded product IDs."""
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump({
                'embedded_ids': list(embedded_ids),
                'last_updated': str(Path().cwd()),
                'model': FREE_EMBEDDING_MODEL,
                'collection': COLLECTION_NAME,
                'api_type': 'free_local'
            }, f, indent=2)


async def run_free_ingestion_pipeline():
    """Run the complete free ingestion pipeline."""
    print("🚀 Starting FREE ingestion pipeline...")
    
    # Load and clean data
    from src.ingest.loader import load_and_clean
    df = load_and_clean()
    print(f"✅ Loaded {len(df)} products from CSV")
    
    # Build documents
    documents = build_documents(df)
    print(f"✅ Built {len(documents)} documents")
    
    # Initialize free embedder
    embedder = FreeEmbedder(use_huggingface=False)
    
    # Embed and upsert
    await embedder.embed_and_upsert(documents)
    
    print("✅ Free ingestion pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_free_ingestion_pipeline())
