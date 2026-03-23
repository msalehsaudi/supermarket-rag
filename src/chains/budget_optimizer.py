"""Budget optimizer chain for finding cost-effective shopping baskets."""

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


BUDGET_OPTIMIZER_PROMPT = """You are a budget-conscious shopping advisor.

USER REQUEST: {user_query}

BUDGET: €{budget}
GOAL: {goal}
DIETARY RESTRICTIONS: {restrictions}

AVAILABLE PRODUCTS (sorted by value):
{retrieved_products}

Build the most cost-effective shopping basket that meets the user's goal.
- Prioritize products with best value per euro for the stated goal
- Do not exceed the budget
- Prefer in-stock items
- Group by category
- Include specific quantities and costs

Return JSON:
{{
  "summary": {{
    "total_cost": float,
    "total_items": int,
    "budget_remaining": float,
    "goal_achievement": str
  }},
  "categories": [
    {{
      "name": str,
      "items": [
        {{
          "name": str,
          "brand": str,
          "quantity": str,
          "unit_cost": float,
          "total_cost": float,
          "value_score": float,
          "reason": str
        }}
      ],
      "category_cost": float
    }}
  ],
  "savings_tips": [str],
  "alternatives": [str]
}}

IMPORTANT:
- Stay strictly within the budget constraint
- Focus on value for money (nutrition/quality per euro)
- Be realistic about quantities needed
- Include practical shopping advice"""


class BudgetOptimizerChain:
    """
    Chain for finding the most cost-effective shopping baskets.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the budget optimizer chain.
        
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
            print(f"Loaded {len(self._all_documents)} documents for budget optimization")
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """
        Format retrieved products for the prompt with value calculations.
        
        Args:
            documents: Retrieved product documents
            
        Returns:
            Formatted product list string
        """
        products = []
        
        for doc in documents:
            meta = doc.metadata
            
            # Calculate value scores based on goal
            value_score = self._calculate_value_score(doc)
            
            product_info = (
                f"{meta['name']} ({meta['brand']}) - "
                f"€{meta['price_eur']:.2f}, "
                f"Value Score: {value_score:.2f}"
            )
            
            if meta.get('is_food'):
                product_info += (
                    f", {meta['calories_per_100g']} kcal/100g, "
                    f"{meta['protein_g_per_100g']}g protein/100g"
                )
            
            if meta.get('weight_kg'):
                product_info += f", {meta['weight_kg']}kg package"
            
            products.append(product_info)
        
        return "\n".join(products)
    
    def _calculate_value_score(self, doc: Document) -> float:
        """
        Calculate a value score for a product.
        
        Args:
            doc: Product document
            
        Returns:
            Value score (higher is better value)
        """
        meta = doc.metadata
        base_score = 1.0
        
        if not meta.get('is_food'):
            return base_score
        
        # Nutrition-based value for food items
        price_per_100g = meta['price_eur'] / (meta.get('weight_kg', 1.0) * 10)
        
        if price_per_100g <= 0:
            return base_score
        
        # Protein value
        protein_value = meta.get('protein_g_per_100g', 0) / price_per_100g
        
        # Calorie value (for energy)
        calorie_value = meta.get('calories_per_100g', 0) / price_per_100g
        
        # Combined score (can be adjusted based on priorities)
        combined_score = (protein_value * 0.6 + calorie_value * 0.4) / 100
        
        return max(base_score, combined_score)
    
    async def _retrieve_products(
        self,
        constraints: Dict[str, Any],
        k: int = 40
    ) -> List[Document]:
        """
        Retrieve relevant products for budget optimization.
        
        Args:
            constraints: User constraints
            k: Number of products to retrieve
            
        Returns:
            List of relevant product documents
        """
        await self._load_documents()
        
        # Build query for budget shopping
        query_parts = ["cheap", "budget", "value"]
        
        if constraints.get('goal'):
            query_parts.append(constraints['goal'])
        
        if constraints.get('restrictions'):
            query_parts.extend(constraints['restrictions'])
        
        query = " ".join(query_parts)
        
        # Build metadata filter
        filter_constraints = {
            "in_stock": True
        }
        
        if constraints.get('max_price_per_item'):
            filter_constraints["max_price"] = constraints['max_price_per_item']
        
        if constraints.get('budget'):
            # Don't consider items more than 20% of total budget
            max_item_price = constraints['budget'] * 0.2
            if "max_price" not in filter_constraints or max_item_price < filter_constraints["max_price"]:
                filter_constraints["max_price"] = max_item_price
        
        where_filter = build_metadata_filter(filter_constraints)
        
        # Retrieve products
        retrieved = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            where_filter=where_filter,
            k=k
        )
        
        # Sort by value score
        retrieved_with_scores = [
            (doc, self._calculate_value_score(doc))
            for doc in retrieved
        ]
        retrieved_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in retrieved_with_scores]
    
    async def optimize_budget(
        self,
        user_query: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generate an optimized shopping basket within budget.
        
        Args:
            user_query: Original user query
            constraints: Extracted constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to optimize budgets."
            return
        
        # Set default values
        budget = constraints.get('budget', 50.0)
        goal = constraints.get('goal', 'general nutrition')
        restrictions = constraints.get('restrictions', [])
        
        # Retrieve relevant products
        products = await self._retrieve_products(constraints)
        
        if not products:
            yield "I couldn't find suitable products within your budget. Please try increasing your budget or adjusting your requirements."
            return
        
        # Format products for prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Generate optimized basket
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": BUDGET_OPTIMIZER_PROMPT.format(
                            user_query=user_query,
                            budget=budget,
                            goal=goal,
                            restrictions=", ".join(restrictions) if restrictions else "none",
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
            yield f"Error optimizing budget: {str(e)}"
    
    async def compare_prices(
        self,
        product_category: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Compare prices across products in a category.
        
        Args:
            product_category: Category to compare
            constraints: User constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to compare prices."
            return
        
        # Retrieve products in category
        await self._load_documents()
        
        filter_constraints = {
            "category": product_category,
            "in_stock": True
        }
        
        where_filter = build_metadata_filter(filter_constraints)
        
        retrieved = await hybrid_retrieve(
            query=product_category,
            documents=self._all_documents,
            where_filter=where_filter,
            k=20
        )
        
        if not retrieved:
            yield f"I couldn't find any products in the {product_category} category."
            return
        
        # Sort by price
        retrieved.sort(key=lambda doc: doc.metadata.get('price_eur', float('inf')))
        
        # Format comparison
        comparison = f"Price comparison for {product_category}:\n\n"
        
        for i, doc in enumerate(retrieved[:10]):
            meta = doc.metadata
            comparison += f"{i+1}. {meta['name']} ({meta['brand']}) - €{meta['price_eur']:.2f}"
            
            if meta.get('weight_kg'):
                price_per_kg = meta['price_eur'] / meta['weight_kg']
                comparison += f" (€{price_per_kg:.2f}/kg)"
            
            if meta.get('rating'):
                comparison += f" - ★{meta['rating']}"
            
            comparison += "\n"
        
        yield comparison


# Convenience functions
async def budget_optimizer_chain(
    user_query: str,
    constraints: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Generate an optimized shopping basket.
    
    Args:
        user_query: Original user query
        constraints: Extracted constraints
        
    Yields:
        Streaming response chunks
    """
    optimizer = BudgetOptimizerChain()
    async for chunk in optimizer.optimize_budget(user_query, constraints):
        yield chunk


if __name__ == "__main__":
    # Test the budget optimizer
    async def test():
        optimizer = BudgetOptimizerChain()
        
        query = "Build me a shopping list for €30 that focuses on high protein"
        constraints = {
            "budget": 30.0,
            "goal": "high protein",
            "restrictions": []
        }
        
        print("Optimizing budget...")
        response_chunks = []
        async for chunk in optimizer.optimize_budget(query, constraints):
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        print("\n\nBudget optimization completed!")
    
    asyncio.run(test())
