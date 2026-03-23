"""Simple embedder using basic TF-IDF or random embeddings (no heavy dependencies)."""

import json
import numpy as np
from pathlib import Path
from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from src.config import (
    CHECKPOINT_PATH, 
    COLLECTION_NAME,
    EMBEDDING_DIM
)
from src.vectorstore.chroma_store import get_collection
from src.ingest.doc_builder import build_documents


class SimpleEmbedder:
    """
    Simple embedder using TF-IDF vectors (no heavy dependencies).
    """
    
    def __init__(self):
        """Initialize simple embedder."""
        self.vectorizer = None
        self.embeddings_cache = {}
    
    def _get_vectorizer(self, documents: List[Document]) -> TfidfVectorizer:
        """Get or create TF-IDF vectorizer."""
        if self.vectorizer is None:
            texts = [doc.page_content for doc in documents]
            self.vectorizer = TfidfVectorizer(
                max_features=EMBEDDING_DIM,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.vectorizer.fit(texts)
            print(f"✅ TF-IDF vectorizer created with {EMBEDDING_DIM} features")
        
        return self.vectorizer
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using TF-IDF.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized. Call _get_vectorizer first.")
        
        embeddings = self.vectorizer.transform(texts).toarray()
        
        # Convert to dense format
        if hasattr(embeddings, 'toarray'):
            embeddings = embeddings.toarray()
        
        return embeddings.tolist()
    
    async def embed_and_upsert(self, documents: List[Document]) -> None:
        """
        Embed documents and upsert to ChromaDB.
        
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
        
        print(f"🔄 Embedding {len(pending_docs)} new documents using TF-IDF")
        
        # Get vectorizer
        vectorizer = self._get_vectorizer(documents)
        
        # Extract texts and embed
        texts = [doc.page_content for doc in pending_docs]
        embeddings = self.embed_batch(texts)
        
        # Prepare for upsert
        metadatas = [doc.metadata for doc in pending_docs]
        ids = [str(doc.metadata['product_id']) for doc in pending_docs]
        
        # Upsert to ChromaDB
        try:
            collection = get_collection()
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(pending_docs), batch_size):
                batch_docs = pending_docs[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                print(f"📦 Processing batch {i // batch_size + 1}/{(len(pending_docs) + batch_size - 1) // batch_size}")
                
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=texts[i:i + batch_size],
                    metadatas=batch_metadatas
                )
                
                # Update checkpoint
                for doc in batch_docs:
                    checkpoint.add(doc.metadata['product_id'])
            
            self.save_checkpoint(checkpoint)
            print(f"✅ TF-IDF embedding completed! Total products: {len(checkpoint)}")
            
        except Exception as e:
            print(f"❌ Error upserting to ChromaDB: {e}")
            raise
    
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
                'model': 'tf-idf',
                'collection': COLLECTION_NAME,
                'api_type': 'free_local'
            }, f, indent=2)


async def run_simple_ingestion_pipeline():
    """Run the simple ingestion pipeline."""
    print("🚀 Starting SIMPLE (TF-IDF) ingestion pipeline...")
    
    # Load and clean data
    from src.ingest.loader import load_and_clean
    df = load_and_clean()
    print(f"✅ Loaded {len(df)} products from CSV")
    
    # Build documents
    documents = build_documents(df)
    print(f"✅ Built {len(documents)} documents")
    
    # Embed and upsert
    embedder = SimpleEmbedder()
    await embedder.embed_and_upsert(documents)
    
    print("✅ Simple ingestion pipeline completed successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_simple_ingestion_pipeline())
