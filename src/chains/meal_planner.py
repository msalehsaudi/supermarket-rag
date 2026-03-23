"""Meal planner chain for generating structured meal plans."""

import asyncio
import json
from typing import Dict, Any, List, AsyncGenerator
import openai
from openai import AsyncOpenAI
from langchain_core.documents import Document

from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.retrieval.query_rewriter import rewrite_with_hyde, extract_constraints
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.reranker import rerank_documents
from src.vectorstore.chroma_store import build_metadata_filter
from src.ingest.doc_builder import build_documents
from src.ingest.loader import load_and_clean


MEAL_PLAN_PROMPT = """You are a professional nutritionist and budget-conscious meal planner.

USER REQUEST: {user_query}

CONSTRAINTS:
- Diet goal: {diet_type}
- Daily calorie target: {calorie_target} kcal  
- Weekly budget: €{budget}
- Dietary restrictions: {restrictions}
- Days: {days}

AVAILABLE SUPERMARKET PRODUCTS (retrieved from database):
{retrieved_products}

Create a {days}-day meal plan using ONLY products from the list above.
For each day, provide breakfast, lunch, dinner, and a snack.

Return as JSON:
{{
  "summary": {{
    "total_cost": float,
    "avg_daily_calories": float,
    "avg_daily_protein": float,
    "total_days": int
  }},
  "days": [
    {{
      "day": 1,
      "meals": {{
        "breakfast": {{
          "products": [{{ "name": str, "quantity": str, "cost": float, "calories": float, "protein": float }}],
          "total_calories": float,
          "total_cost": float
        }},
        "lunch": {{ ... }},
        "dinner": {{ ... }},
        "snack": {{ ... }}
      }},
      "day_total_cost": float,
      "day_total_calories": float
    }}
  ]
}}

IMPORTANT:
- Use ONLY products from the provided list
- Stay within the budget constraint
- Match the diet goal and calorie target as closely as possible
- Include specific quantities and costs
- Ensure nutritional calculations are realistic
- If you cannot meet all constraints perfectly, prioritize staying within budget and using available products"""


class MealPlannerChain:
    """
    Chain for generating structured meal plans based on user constraints and available products.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the meal planner chain.
        
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
            print(f"Loaded {len(self._all_documents)} documents for meal planning")
    
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
            if meta.get('is_food'):  # Only include food products
                product_info = (
                    f"{meta['name']} ({meta['brand']}) - "
                    f"€{meta['price_eur']:.2f}, "
                    f"{meta['calories_per_100g']} kcal/100g, "
                    f"{meta['protein_g_per_100g']}g protein/100g"
                )
                if meta.get('weight_kg'):
                    product_info += f", {meta['weight_kg']}kg package"
                products.append(product_info)
        
        return "\n".join(products)
    
    async def _retrieve_products(
        self, 
        constraints: Dict[str, Any],
        k: int = 50
    ) -> List[Document]:
        """
        Retrieve relevant products for meal planning.
        
        Args:
            constraints: User constraints
            k: Number of products to retrieve
            
        Returns:
            List of relevant product documents
        """
        await self._load_documents()
        
        # Build query for meal planning
        query_parts = []
        if constraints.get('diet_type'):
            query_parts.append(constraints['diet_type'])
        if constraints.get('restrictions'):
            query_parts.extend(constraints['restrictions'])
        
        query = "meal plan " + " ".join(query_parts) if query_parts else "meal plan"
        
        # Build metadata filter
        filter_constraints = {
            "food_only": True,
            "in_stock": True
        }
        
        if constraints.get('max_price_per_item'):
            filter_constraints["max_price"] = constraints['max_price_per_item']
        
        if constraints.get('min_protein'):
            filter_constraints["min_protein"] = constraints['min_protein']
        
        if constraints.get('max_calories'):
            filter_constraints["max_calories"] = constraints['max_calories']
        
        where_filter = build_metadata_filter(filter_constraints)
        
        # Retrieve products
        retrieved = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            where_filter=where_filter,
            k=k
        )
        
        # Rerank for meal planning relevance
        reranked = await rerank_documents(query, retrieved, k=k)
        
        # Filter to food products only
        food_products = [doc for doc in reranked if doc.metadata.get('is_food', False)]
        
        return food_products
    
    async def generate_meal_plan(
        self,
        user_query: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generate a meal plan based on user query and constraints.
        
        Args:
            user_query: Original user query
            constraints: Extracted constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to generate meal plans."
            return
        
        # Set default values
        days = constraints.get('days', 7)
        budget = constraints.get('budget', 60.0)
        diet_type = constraints.get('diet_type', 'balanced')
        calorie_target = constraints.get('calorie_target', 2000)
        restrictions = constraints.get('restrictions', [])
        
        # Retrieve relevant products
        products = await self._retrieve_products(constraints)
        
        if not products:
            yield "I couldn't find suitable food products for your meal plan. Please try adjusting your constraints."
            return
        
        # Format products for prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Generate meal plan
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": MEAL_PLAN_PROMPT.format(
                            user_query=user_query,
                            diet_type=diet_type,
                            calorie_target=calorie_target,
                            budget=budget,
                            restrictions=", ".join(restrictions) if restrictions else "none",
                            days=days,
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
            yield f"Error generating meal plan: {str(e)}"
    
    async def amend_meal_plan(
        self,
        original_plan: Dict[str, Any],
        amendment_request: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Amend an existing meal plan based on user feedback.
        
        Args:
            original_plan: Original meal plan JSON
            amendment_request: User's amendment request
            constraints: User constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to amend meal plans."
            return
        
        AMENDMENT_PROMPT = """You are a nutritionist helping modify an existing meal plan.

ORIGINAL MEAL PLAN:
{original_plan}

USER REQUESTED CHANGE:
{amendment_request}

CONSTRAINTS:
- Budget: €{budget}
- Diet goal: {diet_type}
- Calorie target: {calorie_target} kcal

Please modify the meal plan to address the user's request while staying within constraints.
Return the complete updated meal plan in the same JSON format.
Focus only on the specific changes requested, keep other meals the same."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": AMENDMENT_PROMPT.format(
                            original_plan=json.dumps(original_plan, indent=2),
                            amendment_request=amendment_request,
                            budget=constraints.get('budget', 60.0),
                            diet_type=constraints.get('diet_type', 'balanced'),
                            calorie_target=constraints.get('calorie_target', 2000)
                        )
                    }
                ],
                temperature=self.temperature,
                max_tokens=LLM_MAX_TOKENS,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error amending meal plan: {str(e)}"


# Convenience function
async def generate_meal_plan_chain(
    user_query: str,
    constraints: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Generate a meal plan using the meal planner chain.
    
    Args:
        user_query: Original user query
        constraints: Extracted constraints
        
    Yields:
        Streaming response chunks
    """
    planner = MealPlannerChain()
    async for chunk in planner.generate_meal_plan(user_query, constraints):
        yield chunk


if __name__ == "__main__":
    # Test the meal planner
    async def test():
        planner = MealPlannerChain()
        
        query = "7-day weight loss meal plan with €60 budget"
        constraints = {
            "days": 7,
            "budget": 60.0,
            "diet_type": "weight_loss",
            "calorie_target": 1800
        }
        
        print("Generating meal plan...")
        response_chunks = []
        async for chunk in planner.generate_meal_plan(query, constraints):
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        print("\n\nMeal plan generation completed!")
    
    asyncio.run(test())
