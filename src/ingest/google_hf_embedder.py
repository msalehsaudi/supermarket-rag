"""Embedder using Hugging Face for embeddings."""

import json
import asyncio
from pathlib import Path
from typing import List, Set
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    HF_EMBEDDING_MODEL, 
    HF_EMBEDDING_DIM, 
    EMBEDDING_BATCH_SIZE,
    CHECKPOINT_PATH,
    COLLECTION_NAME
)
from src.vectorstore.chroma_store import get_collection
from src.ingest.doc_builder import build_documents


class GoogleHFEmbedder:
    """
    Embedder using Hugging Face sentence transformers.
    """
    
    def __init__(self):
        """Initialize Hugging Face embedder."""
        try:
            print(f"🔄 Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=HF_EMBEDDING_MODEL,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
            )
            print(f"✅ Hugging Face embeddings loaded successfully")
            print(f"📏 Embedding dimension: {HF_EMBEDDING_DIM}")
        except Exception as e:
            print(f"❌ Error loading Hugging Face embeddings: {e}")
            # Fallback to local
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer(HF_EMBEDDING_MODEL)
            print(f"✅ Fallback to local embeddings: {HF_EMBEDDING_MODEL}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use LangChain HuggingFace embeddings
            embeddings = self.embeddings.embed_documents(texts)
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
        
        print(f"🔄 Embedding {len(pending_docs)} new documents with Hugging Face")
        print(f"⚠️  Skipping {len(documents) - len(pending_docs)} already embedded")
        
        # Extract texts
        texts = [doc.page_content for doc in pending_docs]
        
        # Embed in batches
        all_embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_docs = pending_docs[i:i + batch_size]
            
            print(f"📦 Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} ({len(batch_texts)} documents)")
            
            # Embed batch
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
        
        print(f"🎉 Hugging Face embedding completed! Total products: {len(checkpoint)}")
    
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
                'model': HF_EMBEDDING_MODEL,
                'collection': COLLECTION_NAME,
                'api_type': 'huggingface_google'
            }, f, indent=2)


async def run_google_hf_ingestion_pipeline():
    """Run complete Google/Hugging Face ingestion pipeline."""
    print("🚀 Starting Google/Hugging Face ingestion pipeline...")
    
    # Load and clean data
    from src.ingest.loader import load_and_clean
    df = load_and_clean()
    print(f"✅ Loaded {len(df)} products from CSV")
    
    # Build documents
    documents = build_documents(df)
    print(f"✅ Built {len(documents)} documents")
    
    # Initialize embedder
    embedder = GoogleHFEmbedder()
    
    # Embed and upsert
    await embedder.embed_and_upsert(documents)
    
    print("✅ Google/Hugging Face ingestion pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_google_hf_ingestion_pipeline())
