"""Product search chain for finding and comparing specific products."""

import asyncio
import json
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


PRODUCT_SEARCH_PROMPT = """You are a knowledgeable shopping assistant for a supermarket.

USER REQUEST: {user_query}

SEARCH CRITERIA:
- Category: {category}
- Brand: {brand}
- Price range: €{min_price} - €{max_price}
- Rating: {min_rating}+ stars
- In stock: {in_stock}
- Food only: {food_only}

MATCHING PRODUCTS:
{retrieved_products}

Help the user find the best products for their needs.
Provide detailed comparisons and recommendations.

Return JSON:
{{
  "summary": {{
    "total_matches": int,
    "search_criteria": str,
    "top_recommendation": str
  }},
  "products": [
    {{
      "name": str,
      "brand": str,
      "category": str,
      "price": float,
      "rating": float,
      "in_stock": bool,
      "key_features": [str],
      "pros": [str],
      "cons": [str],
      "best_for": str
    }}
  ],
  "comparisons": [
    {{
      "aspect": str,
      "ranking": [str]
    }}
  ],
  "recommendations": [
    {{
      "use_case": str,
      "product": str,
      "reason": str
    }}
  ],
  "alternatives": [str]
}}

IMPORTANT:
- Use only the product data provided
- Be objective and helpful
- Include practical advice
- Mention if no exact matches are found"""


class ProductSearchChain:
    """
    Chain for finding and comparing specific supermarket products.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the product search chain.
        
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
            print(f"Loaded {len(self._all_documents)} documents for product search")
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """
        Format retrieved products with detailed information.
        
        Args:
            documents: Retrieved product documents
            
        Returns:
            Formatted product list string
        """
        products = []
        
        for doc in documents:
            meta = doc.metadata
            
            product_info = (
                f"{meta['name']} ({meta['brand']})\n"
                f"  Category: {meta['category']}\n"
                f"  Price: €{meta['price_eur']:.2f}"
            )
            
            if meta.get('weight_kg'):
                product_info += f" ({meta['weight_kg']}kg)"
            
            product_info += f"\n  Rating: {meta.get('rating', 0)}/5"
            product_info += f"\n  In stock: {'Yes' if meta.get('in_stock') else 'No'}"
            product_info += f"\n  Origin: {meta.get('origin', 'Unknown')}"
            
            if meta.get('is_food'):
                product_info += (
                    f"\n  Nutrition per 100g: "
                    f"{meta.get('calories_per_100g', 0)} kcal, "
                    f"{meta.get('protein_g_per_100g', 0)}g protein, "
                    f"{meta.get('fat_g_per_100g', 0)}g fat"
                )
            
            products.append(product_info)
        
        return "\n\n".join(products)
    
    async def _retrieve_products(
        self,
        query: str,
        constraints: Dict[str, Any],
        k: int = 25
    ) -> List[Document]:
        """
        Retrieve relevant products for the search query.
        
        Args:
            query: User query
            constraints: User constraints
            k: Number of products to retrieve
            
        Returns:
            List of relevant product documents
        """
        await self._load_documents()
        
        # Build metadata filter
        filter_constraints = {}
        
        if constraints.get('in_stock'):
            filter_constraints["in_stock"] = True
        
        if constraints.get('food_only'):
            filter_constraints["food_only"] = True
        
        if constraints.get('category'):
            filter_constraints["category"] = constraints['category']
        
        if constraints.get('brand'):
            filter_constraints["brand"] = constraints['brand']
        
        if constraints.get('min_price'):
            filter_constraints["min_price"] = constraints['min_price']
        
        if constraints.get('max_price'):
            filter_constraints["max_price"] = constraints['max_price']
        
        if constraints.get('min_rating'):
            filter_constraints["min_rating"] = constraints['min_rating']
        
        where_filter = build_metadata_filter(filter_constraints)
        
        # Retrieve products
        retrieved = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            where_filter=where_filter,
            k=k
        )
        
        # Rerank for search relevance
        reranked = await rerank_documents(query, retrieved, k=k)
        
        return reranked
    
    async def search_products(
        self,
        user_query: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Search for products based on user query and constraints.
        
        Args:
            user_query: Original user query
            constraints: Extracted constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to search for products."
            return
        
        # Retrieve relevant products
        products = await self._retrieve_products(user_query, constraints)
        
        if not products:
            yield "I couldn't find any products matching your criteria. Please try adjusting your search terms or constraints."
            return
        
        # Format products for prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Extract search criteria for prompt
        search_criteria = {
            "category": constraints.get('category', 'any'),
            "brand": constraints.get('brand', 'any'),
            "min_price": constraints.get('min_price', 0),
            "max_price": constraints.get('max_price', 'no limit'),
            "min_rating": constraints.get('min_rating', 'any'),
            "in_stock": constraints.get('in_stock', 'no preference'),
            "food_only": constraints.get('food_only', 'no preference')
        }
        
        # Generate search results
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": PRODUCT_SEARCH_PROMPT.format(
                            user_query=user_query,
                            **search_criteria,
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
            yield f"Error searching products: {str(e)}"
    
    async def get_similar_products(
        self,
        product_name: str,
        k: int = 10
    ) -> AsyncGenerator[str, None]:
        """
        Find products similar to a given product.
        
        Args:
            product_name: Name of the reference product
            k: Number of similar products to find
            
        Yields:
            Streaming response chunks
        """
        await self._load_documents()
        
        # Find the reference product
        reference_doc = None
        for doc in self._all_documents:
            if (doc.metadata.get('name', '').lower() == product_name.lower() or
                product_name.lower() in doc.metadata.get('name', '').lower()):
                reference_doc = doc
                break
        
        if not reference_doc:
            yield f"I couldn't find a product named '{product_name}'. Please check the spelling or try a different product."
            return
        
        # Use the product description as query for similarity search
        query = reference_doc.page_content
        
        # Retrieve similar products (excluding the original)
        similar_docs = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            k=k + 1  # Get one extra to exclude the original
        )
        
        # Filter out the original product
        similar_products = [
            doc for doc in similar_docs 
            if doc.metadata.get('product_id') != reference_doc.metadata.get('product_id')
        ][:k]
        
        if not similar_products:
            yield f"I couldn't find products similar to '{product_name}'."
            return
        
        # Format results
        response = f"Products similar to {reference_doc.metadata.get('name')}:\n\n"
        
        for i, doc in enumerate(similar_products):
            meta = doc.metadata
            response += f"{i+1}. {meta['name']} ({meta['brand']})\n"
            response += f"   Category: {meta['category']}\n"
            response += f"   Price: €{meta['price_eur']:.2f}\n"
            response += f"   Rating: {meta.get('rating', 0)}/5\n"
            
            if meta.get('is_food'):
                response += f"   Nutrition: {meta.get('calories_per_100g', 0)} kcal, {meta.get('protein_g_per_100g', 0)}g protein\n"
            
            response += "\n"
        
        yield response
    
    async def get_category_overview(
        self,
        category: str
    ) -> AsyncGenerator[str, None]:
        """
        Get an overview of products in a specific category.
        
        Args:
            category: Product category
            
        Yields:
            Streaming response chunks
        """
        await self._load_documents()
        
        # Filter by category
        category_docs = [
            doc for doc in self._all_documents 
            if doc.metadata.get('category', '').lower() == category.lower()
        ]
        
        if not category_docs:
            yield f"I couldn't find any products in the '{category}' category."
            return
        
        # Sort by rating and price
        category_docs.sort(key=lambda doc: (
            doc.metadata.get('rating', 0), 
            -doc.metadata.get('price_eur', float('inf'))
        ), reverse=True)
        
        # Generate overview
        response = f"Overview of {category} category:\n\n"
        response += f"Total products: {len(category_docs)}\n\n"
        
        # Top rated products
        response += "Top rated products:\n"
        for doc in category_docs[:5]:
            meta = doc.metadata
            response += f"• {meta['name']} ({meta['brand']}) - ★{meta.get('rating', 0)}/5, €{meta['price_eur']:.2f}\n"
        
        response += "\nPrice range: "
        prices = [doc.metadata.get('price_eur', 0) for doc in category_docs]
        response += f"€{min(prices):.2f} - €{max(prices):.2f}\n"
        
        # Average rating
        ratings = [doc.metadata.get('rating', 0) for doc in category_docs if doc.metadata.get('rating', 0) > 0]
        if ratings:
            response += f"Average rating: {sum(ratings)/len(ratings):.1f}/5\n"
        
        yield response


# Convenience function
async def product_search_chain(
    user_query: str,
    constraints: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Search for products based on user query.
    
    Args:
        user_query: Original user query
        constraints: Extracted constraints
        
    Yields:
        Streaming response chunks
    """
    searcher = ProductSearchChain()
    async for chunk in searcher.search_products(user_query, constraints):
        yield chunk


if __name__ == "__main__":
    # Test the product search
    async def test():
        searcher = ProductSearchChain()
        
        query = "Italian cheeses with rating above 4"
        constraints = {
            "category": "Dairy & Eggs",
            "min_rating": 4.0
        }
        
        print("Searching for products...")
        response_chunks = []
        async for chunk in searcher.search_products(query, constraints):
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        print("\n\nProduct search completed!")
    
    asyncio.run(test())
