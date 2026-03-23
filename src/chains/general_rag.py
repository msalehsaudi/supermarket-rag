"""General RAG chain for handling any query with product information."""

import asyncio
from typing import Dict, Any, List, AsyncGenerator
import openai
from openai import AsyncOpenAI
from langchain_core.documents import Document

from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.reranker import rerank_documents
from src.vectorstore.chroma_store import build_metadata_filter
from src.ingest.doc_builder import build_documents
from src.ingest.loader import load_and_clean


GENERAL_RAG_PROMPT = """You are a helpful supermarket assistant.

USER QUESTION: {user_query}

RELEVANT PRODUCTS:
{retrieved_products}

Answer the user's question using the product information provided.
Be helpful, accurate, and conversational.
If the products don't fully answer the question, explain what information is available and what might be missing.

Guidelines:
- Use specific product details when relevant
- Mention prices, ratings, and other attributes when helpful
- Be honest about limitations in the available data
- Provide practical advice when appropriate
- Keep responses concise but comprehensive"""

SIMPLE_SEARCH_PROMPT = """You are a supermarket search assistant.

USER QUERY: {user_query}

RELEVANT PRODUCTS:
{retrieved_products}

Help the user with their query using the product information above.
Focus on being helpful and direct."""


class GeneralRAGChain:
    """
    General RAG chain for handling any type of query about supermarket products.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the general RAG chain.
        
        Args:
            model: LLM model to use
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self._all_documents = None
    
    async def _load_documents(self):
        """Load all documents for retrieval."""
        if self._all_documents is None:
            df = load_and_clean()
            self._all_documents = build_documents(df)
            print(f"Loaded {len(self._all_documents)} documents for general RAG")
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """
        Format retrieved products for the prompt.
        
        Args:
            documents: Retrieved product documents
            
        Returns:
            Formatted product list string
        """
        products = []
        
        for doc in documents:
            meta = doc.metadata
            
            product_info = (
                f"{meta['name']} ({meta['brand']}) - "
                f"{meta['category']}, "
                f"€{meta['price_eur']:.2f}"
            )
            
            if meta.get('weight_kg'):
                product_info += f" ({meta['weight_kg']}kg)"
            
            product_info += f", Rating: {meta.get('rating', 0)}/5"
            
            if meta.get('is_food'):
                product_info += (
                    f", Nutrition: {meta.get('calories_per_100g', 0)} kcal, "
                    f"{meta.get('protein_g_per_100g', 0)}g protein"
                )
            
            product_info += f", Stock: {'In stock' if meta.get('in_stock') else 'Out of stock'}"
            
            products.append(product_info)
        
        return "\n".join(products)
    
    async def _retrieve_products(
        self,
        query: str,
        constraints: Dict[str, Any],
        k: int = 15
    ) -> List[Document]:
        """
        Retrieve relevant products for the query.
        
        Args:
            query: User query
            constraints: User constraints
            k: Number of products to retrieve
            
        Returns:
            List of relevant product documents
        """
        await self._load_documents()
        
        # Build metadata filter (minimal for general queries)
        filter_constraints = {}
        
        if constraints.get('in_stock'):
            filter_constraints["in_stock"] = True
        
        if constraints.get('food_only'):
            filter_constraints["food_only"] = True
        
        if constraints.get('category'):
            filter_constraints["category"] = constraints['category']
        
        if constraints.get('max_price'):
            filter_constraints["max_price"] = constraints['max_price']
        
        where_filter = build_metadata_filter(filter_constraints)
        
        # Retrieve products
        retrieved = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            where_filter=where_filter,
            k=k
        )
        
        # Rerank for relevance
        reranked = await rerank_documents(query, retrieved, k=k)
        
        return reranked
    
    async def answer_query(
        self,
        user_query: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Answer a general query using RAG.
        
        Args:
            user_query: Original user query
            constraints: Extracted constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to answer questions about products."
            return
        
        # Retrieve relevant products
        products = await self._retrieve_products(user_query, constraints)
        
        # Choose prompt based on query type
        if len(products) == 0:
            yield "I couldn't find any relevant products for your query. Please try different search terms or check if the products you're looking for are available in our database."
            return
        
        # Format products for prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Use simpler prompt for basic queries
        prompt_template = SIMPLE_SEARCH_PROMPT if len(user_query.split()) < 8 else GENERAL_RAG_PROMPT
        
        # Generate response
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            user_query=user_query,
                            retrieved_products=formatted_products
                        )
                    }
                ],
                temperature=self.temperature,
                max_tokens=LLM_MAX_TOKENS,
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error answering query: {str(e)}"
    
    async def get_store_info(self) -> AsyncGenerator[str, None]:
        """
        Get general information about the store/product database.
        
        Yields:
            Streaming response chunks
        """
        await self._load_documents()
        
        # Calculate statistics
        total_products = len(self._all_documents)
        food_products = sum(1 for doc in self._all_documents if doc.metadata.get('is_food', False))
        in_stock = sum(1 for doc in self._all_documents if doc.metadata.get('in_stock', False))
        
        categories = set(doc.metadata.get('category', 'Unknown') for doc in self._all_documents)
        brands = set(doc.metadata.get('brand', 'Unknown') for doc in self._all_documents)
        
        price_range = [
            doc.metadata.get('price_eur', 0) 
            for doc in self._all_documents 
            if doc.metadata.get('price_eur', 0) > 0
        ]
        
        response = f"""🛒 Supermarket Database Information:

📊 Statistics:
• Total products: {total_products:,}
• Food products: {food_products:,}
• In stock: {in_stock:,}
• Categories: {len(categories)}
• Brands: {len(brands)}

💰 Price range: €{min(price_range):.2f} - €{max(price_range):.2f}

📦 Top categories:
"""
        
        # Count products per category
        category_counts = {}
        for doc in self._all_documents:
            cat = doc.metadata.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Show top 10 categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for cat, count in top_categories:
            response += f"• {cat}: {count} products\n"
        
        response += f"""
🔍 I can help you with:
• Finding specific products
• Comparing prices and nutrition
• Creating meal plans
• Budget optimization
• Nutrition advice

Feel free to ask me anything about our products!"""
        
        yield response


# Convenience function
async def general_rag_chain(
    user_query: str,
    constraints: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Answer general queries using RAG.
    
    Args:
        user_query: Original user query
        constraints: Extracted constraints
        
    Yields:
        Streaming response chunks
    """
    rag = GeneralRAGChain()
    async for chunk in rag.answer_query(user_query, constraints):
        yield chunk


if __name__ == "__main__":
    # Test the general RAG chain
    async def test():
        rag = GeneralRAGChain()
        
        queries = [
            "What products do you have?",
            "Tell me about your dairy products",
            "What's the most expensive item?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            print("Response: ", end="")
            async for chunk in rag.answer_query(query, {}):
                print(chunk, end="", flush=True)
            print()
    
    asyncio.run(test())
