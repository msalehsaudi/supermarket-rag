"""Cross-encoder reranker for improving retrieval precision."""

import asyncio
from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config import RERANK_TOP_K, RERANK_MODEL


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.
    More accurate than bi-encoder cosine similarity but slower.
    """
    
    def __init__(self, model_name: str = RERANK_MODEL):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            print(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            print(f"Error loading cross-encoder model {self.model_name}: {e}")
            print("Falling back to simple scoring")
            self.model = None
    
    def _simple_score(self, query: str, document: str) -> float:
        """
        Simple scoring fallback when model is not available.
        Uses basic keyword matching.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Simple relevance score
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Jaccard similarity
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _batch_score(
        self, 
        query: str, 
        documents: List[Document],
        batch_size: int = 16
    ) -> List[float]:
        """
        Score documents in batches.
        
        Args:
            query: Query text
            documents: List of documents to score
            batch_size: Batch size for processing
            
        Returns:
            List of relevance scores
        """
        if self.model is None:
            # Use simple scoring
            return [self._simple_score(query, doc.page_content) for doc in documents]
        
        scores = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Prepare pairs for the model
            pairs = [(query, doc.page_content) for doc in batch_docs]
            
            # Get scores from cross-encoder
            batch_scores = self.model.predict(pairs)
            scores.extend(batch_scores.tolist())
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = RERANK_TOP_K
    ) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Score all documents
        scores = self._batch_score(query, documents)
        
        # Pair documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, _ in doc_scores[:k]]
    
    async def arerank(
        self,
        query: str,
        documents: List[Document],
        k: int = RERANK_TOP_K
    ) -> List[Document]:
        """
        Async version of rerank.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, k)


class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) diversifier.
    Balances relevance and diversity in results.
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR diversifier.
        
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
    
    def mmr_score(
        self,
        doc: Document,
        selected_docs: List[Document],
        relevance_scores: List[float],
        doc_idx: int
    ) -> float:
        """
        Calculate MMR score for a document.
        
        Args:
            doc: Document to score
            selected_docs: Already selected documents
            relevance_scores: Relevance scores for all documents
            doc_idx: Index of the document in relevance_scores
            
        Returns:
            MMR score
        """
        relevance = relevance_scores[doc_idx]
        
        if not selected_docs:
            return relevance
        
        # Calculate maximum similarity to selected documents
        max_similarity = 0.0
        
        for selected_doc in selected_docs:
            # Simple similarity using word overlap
            doc_words = set(doc.page_content.lower().split())
            selected_words = set(selected_doc.page_content.lower().split())
            
            intersection = doc_words.intersection(selected_words)
            union = doc_words.union(selected_words)
            
            if union:
                similarity = len(intersection) / len(union)
                max_similarity = max(max_similarity, similarity)
        
        # MMR score: λ * relevance - (1-λ) * max_similarity
        return self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
    
    def diversify(
        self,
        query: str,
        documents: List[Document],
        relevance_scores: List[float],
        k: int = RERANK_TOP_K
    ) -> List[Document]:
        """
        Diversify results using MMR.
        
        Args:
            query: Query text (not used in current implementation)
            documents: List of documents to diversify
            relevance_scores: Relevance scores for documents
            k: Number of documents to return
            
        Returns:
            Diversified list of documents
        """
        if not documents or len(documents) <= k:
            return documents
        
        selected_docs = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (highest relevance)
        first_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
        selected_docs.append(documents[first_idx])
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        while len(selected_docs) < k and remaining_indices:
            # Find document with highest MMR score
            best_idx = max(
                remaining_indices,
                key=lambda i: self.mmr_score(
                    documents[i], selected_docs, relevance_scores, i
                )
            )
            
            selected_docs.append(documents[best_idx])
            remaining_indices.remove(best_idx)
        
        return selected_docs


# Convenience functions
async def rerank_documents(
    query: str,
    documents: List[Document],
    k: int = RERANK_TOP_K,
    model_name: str = RERANK_MODEL
) -> List[Document]:
    """
    Rerank documents using cross-encoder.
    
    Args:
        query: Query text
        documents: List of documents to rerank
        k: Number of top documents to return
        model_name: Cross-encoder model name
        
    Returns:
        Reranked list of documents
    """
    reranker = CrossEncoderReranker(model_name)
    return await reranker.arerank(query, documents, k)


def diversify_documents(
    query: str,
    documents: List[Document],
    relevance_scores: List[float],
    k: int = RERANK_TOP_K,
    lambda_param: float = 0.7
) -> List[Document]:
    """
    Diversify documents using MMR.
    
    Args:
        query: Query text
        documents: List of documents to diversify
        relevance_scores: Relevance scores for documents
        k: Number of documents to return
        lambda_param: MMR lambda parameter
        
    Returns:
        Diversified list of documents
    """
    diversifier = MMRDiversifier(lambda_param)
    return diversifier.diversify(query, documents, relevance_scores, k)


if __name__ == "__main__":
    # Test the reranker
    import asyncio
    from src.ingest.loader import load_and_clean
    from src.ingest.doc_builder import build_documents
    
    async def test():
        # Load sample data
        df = load_and_clean()
        documents = build_documents(df[:50])  # Test with first 50 products
        
        # Test reranking
        query = "high protein chicken breast"
        
        # Get initial results (simulated)
        initial_docs = documents[:20]  # Pretend these are retrieved
        
        # Rerank
        reranked = await rerank_documents(query, initial_docs, k=5)
        
        print(f"Query: {query}")
        print(f"Reranked to {len(reranked)} results:")
        for i, doc in enumerate(reranked):
            print(f"{i+1}. {doc.metadata.get('name', 'Unknown')} - {doc.metadata.get('category', 'Unknown')}")
    
    asyncio.run(test())
