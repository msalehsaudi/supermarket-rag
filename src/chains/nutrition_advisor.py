"""Nutrition advisor chain for answering nutrition-related queries."""

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


NUTRITION_QUERY_PROMPT = """You are a registered nutritionist and food science expert.

USER QUESTION: {user_query}

NUTRITION FOCUS: {nutrition_focus}

AVAILABLE PRODUCTS (relevant to the query):
{retrieved_products}

Answer the user's nutrition question using ONLY the product data provided.
Include specific product examples with their nutritional information.
Provide practical, evidence-based advice.

Return JSON:
{{
  "answer": str,
  "key_findings": [str],
  "product_recommendations": [
    {{
      "name": str,
      "brand": str,
      "reason": str,
      "nutrition_highlight": str,
      "price": float
    }}
  ],
  "general_advice": [str],
  "warnings": [str]
}}

IMPORTANT:
- Base answers only on the provided product data
- Include specific nutritional values from the products
- Be scientifically accurate and practical
- Mention any limitations or considerations
- If data is insufficient, clearly state that"""


class NutritionAdvisorChain:
    """
    Chain for answering nutrition-related questions about supermarket products.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the nutrition advisor chain.
        
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
            print(f"Loaded {len(self._all_documents)} documents for nutrition advice")
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """
        Format retrieved products with detailed nutrition information.
        
        Args:
            documents: Retrieved product documents
            
        Returns:
            Formatted product list string
        """
        products = []
        
        for doc in documents:
            meta = doc.metadata
            
            if not meta.get('is_food'):
                continue
            
            nutrition_info = (
                f"{meta['name']} ({meta['brand']}) - "
                f"€{meta['price_eur']:.2f}\n"
                f"  Nutrition per 100g: "
                f"{meta['calories_per_100g']} kcal, "
                f"{meta['protein_g_per_100g']}g protein, "
                f"{meta['fat_g_per_100g']}g fat, "
                f"{meta['carbs_g_per_100g']}g carbs, "
                f"{meta['sugar_g_per_100g']}g sugar, "
                f"{meta['fiber_g_per_100g']}g fiber, "
                f"{meta['sodium_mg_per_100g']}mg sodium"
            )
            
            if meta.get('weight_kg'):
                nutrition_info += f"\n  Package size: {meta['weight_kg']}kg"
            
            products.append(nutrition_info)
        
        return "\n\n".join(products)
    
    def _extract_nutrition_focus(self, query: str) -> str:
        """
        Extract the main nutrition focus from the query.
        
        Args:
            query: User query
            
        Returns:
            Nutrition focus string
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["protein", "proteins"]):
            return "protein content"
        elif any(word in query_lower for word in ["calories", "calorie", "energy"]):
            return "calories"
        elif any(word in query_lower for word in ["fat", "fats", "lipid"]):
            return "fat content"
        elif any(word in query_lower for word in ["carb", "carbs", "carbohydrate"]):
            return "carbohydrates"
        elif any(word in query_lower for word in ["sugar", "sugars"]):
            return "sugar content"
        elif any(word in query_lower for word in ["fiber", "fibers"]):
            return "fiber content"
        elif any(word in query_lower for word in ["sodium", "salt"]):
            return "sodium content"
        else:
            return "general nutrition"
    
    async def _retrieve_products(
        self,
        query: str,
        constraints: Dict[str, Any],
        k: int = 30
    ) -> List[Document]:
        """
        Retrieve relevant products for nutrition query.
        
        Args:
            query: User query
            constraints: User constraints
            k: Number of products to retrieve
            
        Returns:
            List of relevant product documents
        """
        await self._load_documents()
        
        # Build metadata filter
        filter_constraints = {
            "food_only": True,
            "in_stock": True
        }
        
        # Add nutrition-specific filters
        if constraints.get('min_protein'):
            filter_constraints["min_protein"] = constraints['min_protein']
        
        if constraints.get('max_calories'):
            filter_constraints["max_calories"] = constraints['max_calories']
        
        if constraints.get('max_sugar'):
            filter_constraints["max_sugar"] = constraints['max_sugar']
        
        if constraints.get('max_price_per_item'):
            filter_constraints["max_price"] = constraints['max_price_per_item']
        
        where_filter = build_metadata_filter(filter_constraints)
        
        # Retrieve products
        retrieved = await hybrid_retrieve(
            query=query,
            documents=self._all_documents,
            where_filter=where_filter,
            k=k
        )
        
        # Rerank for nutrition relevance
        reranked = await rerank_documents(query, retrieved, k=k)
        
        # Filter to food products only
        food_products = [doc for doc in reranked if doc.metadata.get('is_food', False)]
        
        return food_products
    
    async def answer_nutrition_query(
        self,
        user_query: str,
        constraints: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Answer a nutrition-related question.
        
        Args:
            user_query: Original user query
            constraints: Extracted constraints
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to provide nutrition advice."
            return
        
        # Extract nutrition focus
        nutrition_focus = self._extract_nutrition_focus(user_query)
        
        # Retrieve relevant products
        products = await self._retrieve_products(user_query, constraints)
        
        if not products:
            yield "I couldn't find relevant food products to answer your nutrition question. Please try a more specific query."
            return
        
        # Format products for prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Generate nutrition advice
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": NUTRITION_QUERY_PROMPT.format(
                            user_query=user_query,
                            nutrition_focus=nutrition_focus,
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
            yield f"Error answering nutrition query: {str(e)}"
    
    async def compare_nutrition(
        self,
        products: List[str],
        nutrition_metric: str
    ) -> AsyncGenerator[str, None]:
        """
        Compare nutrition between specific products.
        
        Args:
            products: List of product names to compare
            nutrition_metric: Nutrition metric to compare
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            yield "Sorry, I need an OpenAI API key to compare nutrition."
            return
        
        await self._load_documents()
        
        # Find specific products
        found_products = []
        for product_name in products:
            for doc in self._all_documents:
                if (doc.metadata.get('name', '').lower() == product_name.lower() or
                    product_name.lower() in doc.metadata.get('name', '').lower()):
                    found_products.append(doc)
                    break
        
        if len(found_products) < 2:
            yield f"I could only find {len(found_products)} of the requested products. Please check the product names."
            return
        
        # Create comparison
        comparison = f"Nutrition comparison ({nutrition_metric}):\n\n"
        
        for doc in found_products:
            meta = doc.metadata
            comparison += f"{meta['name']} ({meta['brand']}):\n"
            
            if nutrition_metric.lower() in ["protein", "proteins"]:
                comparison += f"  Protein: {meta.get('protein_g_per_100g', 0)}g per 100g\n"
            elif nutrition_metric.lower() in ["calories", "calorie", "energy"]:
                comparison += f"  Calories: {meta.get('calories_per_100g', 0)} kcal per 100g\n"
            elif nutrition_metric.lower() in ["fat", "fats"]:
                comparison += f"  Fat: {meta.get('fat_g_per_100g', 0)}g per 100g\n"
            elif nutrition_metric.lower() in ["sugar", "sugars"]:
                comparison += f"  Sugar: {meta.get('sugar_g_per_100g', 0)}g per 100g\n"
            else:
                # Full nutrition breakdown
                comparison += (
                    f"  Calories: {meta.get('calories_per_100g', 0)} kcal\n"
                    f"  Protein: {meta.get('protein_g_per_100g', 0)}g\n"
                    f"  Fat: {meta.get('fat_g_per_100g', 0)}g\n"
                    f"  Carbs: {meta.get('carbs_g_per_100g', 0)}g\n"
                    f"  Sugar: {meta.get('sugar_g_per_100g', 0)}g\n"
                    f"  Fiber: {meta.get('fiber_g_per_100g', 0)}g\n"
                )
            
            comparison += f"  Price: €{meta.get('price_eur', 0):.2f}\n\n"
        
        yield comparison


# Convenience function
async def nutrition_advisor_chain(
    user_query: str,
    constraints: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Answer nutrition-related questions.
    
    Args:
        user_query: Original user query
        constraints: Extracted constraints
        
    Yields:
        Streaming response chunks
    """
    advisor = NutritionAdvisorChain()
    async for chunk in advisor.answer_nutrition_query(user_query, constraints):
        yield chunk


if __name__ == "__main__":
    # Test the nutrition advisor
    async def test():
        advisor = NutritionAdvisorChain()
        
        query = "Which products have the highest protein content under €5?"
        constraints = {
            "max_price_per_item": 5.0
        }
        
        print("Answering nutrition query...")
        response_chunks = []
        async for chunk in advisor.answer_nutrition_query(query, constraints):
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        print("\n\nNutrition advice completed!")
    
    asyncio.run(test())
