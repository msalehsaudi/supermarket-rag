"""Hybrid retriever combining vector search and BM25."""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from src.vectorstore.chroma_store import get_collection, query_collection
from src.config import RETRIEVAL_TOP_K


class HybridRetriever:
    """
    Hybrid retriever that combines vector similarity search with BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """
    
    def __init__(self, vector_weight: float = 0.6, bm25_weight: float = 0.4):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_weight: Weight for vector search results (default: 0.6)
            bm25_weight: Weight for BM25 results (default: 0.4)
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.collection = get_collection()
        self._bm25_index = None
        self._documents = None
    
    def _build_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            BM25 index
        """
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = []
        for doc in documents:
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        return BM25Okapi(tokenized_docs)
    
    async def _vector_search(
        self, 
        query: str, 
        where_filter: Optional[Dict[str, Any]] = None,
        k: int = RETRIEVAL_TOP_K
    ) -> List[tuple[Document, float]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            where_filter: Metadata filter
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        results = query_collection(
            query_text=query,
            where_filter=where_filter,
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            # Convert distance to similarity score (lower distance = higher similarity)
            score = 1.0 / (1.0 + results['distances'][0][i])
            documents.append((doc, score))
        
        return documents
    
    def _bm25_search(
        self, 
        query: str,
        documents: List[Document],
        k: int = RETRIEVAL_TOP_K
    ) -> List[tuple[Document, float]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            documents: List of all documents to search within
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        # Build BM25 index if not exists
        if self._bm25_index is None or self._documents != documents:
            self._bm25_index = self._build_bm25_index(documents)
            self._documents = documents
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Create list of (document, score) tuples
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending) and take top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:k]
    
    def reciprocal_rank_fusion(
        self, 
        vector_results: List[tuple[Document, float]],
        bm25_results: List[tuple[Document, float]],
        k: int = 60  # RRF constant
    ) -> List[tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default: 60)
            
        Returns:
            Fused list of (document, score) tuples
        """
        # Create document ID to score mapping
        doc_scores = {}
        
        # Add vector search results
        for rank, (doc, score) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get('product_id', id(doc))
            rrf_score = self.vector_weight * (1.0 / (k + rank))
            doc_scores[doc_id] = rrf_score
        
        # Add BM25 results
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get('product_id', id(doc))
            rrf_score = self.bm25_weight * (1.0 / (k + rank))
            
            # If document already exists, add scores
            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
            else:
                doc_scores[doc_id] = rrf_score
        
        # Convert back to document list
        doc_map = {doc.metadata.get('product_id', id(doc)): doc for doc, _ in vector_results}
        doc_map.update({doc.metadata.get('product_id', id(doc)): doc for doc, _ in bm25_results})
        
        # Sort by fused score
        fused_results = [
            (doc_map[doc_id], score) 
            for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return fused_results
    
    async def retrieve(
        self,
        query: str,
        documents: List[Document],
        where_filter: Optional[Dict[str, Any]] = None,
        k: int = RETRIEVAL_TOP_K
    ) -> List[Document]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: Search query
            documents: List of all documents to search within
            where_filter: Metadata filter for vector search
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Perform both searches in parallel
        vector_task = self._vector_search(query, where_filter, k)
        bm25_task = asyncio.get_event_loop().run_in_executor(
            None, self._bm25_search, query, documents, k
        )
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # Fuse results
        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Return only documents (without scores)
        return [doc for doc, _ in fused_results[:k]]


# Convenience function for quick usage
async def hybrid_retrieve(
    query: str,
    documents: List[Document],
    where_filter: Optional[Dict[str, Any]] = None,
    k: int = RETRIEVAL_TOP_K,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> List[Document]:
    """
    Perform hybrid retrieval with default settings.
    
    Args:
        query: Search query
        documents: List of all documents to search within
        where_filter: Metadata filter for vector search
        k: Number of results to return
        vector_weight: Weight for vector search results
        bm25_weight: Weight for BM25 results
        
    Returns:
        List of retrieved documents
    """
    retriever = HybridRetriever(vector_weight, bm25_weight)
    return await retriever.retrieve(query, documents, where_filter, k)


if __name__ == "__main__":
    # Test the hybrid retriever
    import asyncio
    from src.ingest.loader import load_and_clean
    from src.ingest.doc_builder import build_documents
    
    async def test():
        # Load sample data
        df = load_and_clean()
        documents = build_documents(df[:100])  # Test with first 100 products
        
        # Test retrieval
        query = "high protein chicken breast"
        results = await hybrid_retrieve(query, documents)
        
        print(f"Query: {query}")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results[:5]):
            print(f"{i+1}. {doc.metadata.get('name', 'Unknown')} - {doc.metadata.get('category', 'Unknown')}")
    
    asyncio.run(test())
