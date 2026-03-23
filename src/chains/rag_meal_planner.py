"""RAG-enabled meal planner using database retrieval + Google Gemini."""

import asyncio
from typing import Dict, Any, AsyncGenerator, List
from langchain_core.documents import Document

from src.config import get_llm_config
from src.retrieval.query_rewriter import rewrite_with_hyde, extract_constraints
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.reranker import rerank_documents
from src.vectorstore.chroma_store import build_metadata_filter
from src.ingest.doc_builder import build_documents
from src.ingest.loader import load_and_clean


class RAGMealPlanner:
    """Meal planner that retrieves products from database before generating plans."""
    
    def __init__(self):
        """Initialize RAG meal planner."""
        self.config = get_llm_config()
        self._all_documents = None
        
        # Initialize Google Gemini
        import google.generativeai as genai
        genai.configure(api_key=self.config["api_key"])
        self.model = genai.GenerativeModel(self.config["model"])
        print(f"🤖 RAG Meal Planner initialized: {self.config['model']}")
    
    async def _load_documents(self):
        """Load all documents for retrieval."""
        if self._all_documents is None:
            df = load_and_clean()
            self._all_documents = build_documents(df)
            print(f"📚 Loaded {len(self._all_documents)} products for retrieval")
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """Format retrieved products for the prompt."""
        products = []
        for doc in documents[:20]:  # Limit to top 20 products
            meta = doc.metadata
            product_info = f"- {meta['name']} ({meta['brand']}) - €{meta['price_eur']:.2f}"
            
            if meta.get('is_food'):
                nutrition = []
                if meta.get('calories_per_100g'):
                    nutrition.append(f"{meta['calories_per_100g']} kcal")
                if meta.get('protein_g_per_100g'):
                    nutrition.append(f"{meta['protein_g_per_100g']}g protein")
                if nutrition:
                    product_info += f" [{', '.join(nutrition)} per 100g]"
            
            products.append(product_info)
        
        return "\n".join(products)
    
    async def _retrieve_products(
        self, 
        constraints: Dict[str, Any],
        k: int = 50
    ) -> List[Document]:
        """Retrieve relevant products for meal planning."""
        await self._load_documents()
        
        # Build query for retrieval
        query_parts = ["meal planning", "food products"]
        if constraints.get("diet_type"):
            query_parts.append(constraints["diet_type"])
        if constraints.get("budget"):
            query_parts.append(f"budget {constraints['budget']}")
        
        query = " ".join(query_parts)
        
        # Build filter constraints
        filter_constraints = {"is_food": True}
        if constraints.get("diet_type") == "vegan":
            filter_constraints["category"] = ["plant-based", "vegetables", "fruits", "grains"]
        elif constraints.get("diet_type") == "vegetarian":
            filter_constraints["category"] = ["plant-based", "vegetables", "fruits", "grains", "dairy"]
        
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
        
        # Filter to food products only and apply budget constraints
        food_products = []
        budget = constraints.get("budget", float('inf'))
        
        for doc in reranked:
            if doc.metadata.get('is_food', False):
                price = doc.metadata.get('price_eur', float('inf'))
                if price <= budget * 0.1:  # Allow items up to 10% of total budget
                    food_products.append(doc)
        
        return food_products[:20]  # Return top 20 relevant products
    
    async def generate_meal_plan(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate meal plan using retrieved products + Gemini."""
        # Retrieve relevant products from database
        products = await self._retrieve_products(constraints)
        
        if not products:
            yield "I couldn't find suitable food products in our database for your meal plan. Please try adjusting your constraints or budget."
            return
        
        # Format products for the prompt
        formatted_products = self._format_products_for_prompt(products)
        
        # Extract constraints
        days = constraints.get("days", 7)
        budget = constraints.get("budget", 50.0)
        diet_type = constraints.get("diet_type", "balanced")
        restrictions = constraints.get("restrictions", [])
        
        # Build prompt with retrieved products
        prompt = f"""Create a detailed {days}-day meal plan for a {diet_type} diet with a €{budget:.2f} budget.

User request: "{user_query}"
- Dietary restrictions: {restrictions}
- Days: {days}

AVAILABLE SUPERMARKET PRODUCTS (retrieved from database):
{formatted_products}

Create a {days}-day meal plan using ONLY products from the list above.
For each day, provide breakfast, lunch, dinner, and a snack.
Use real prices from the product list.
Make sure the total cost stays within €{budget:.2f} budget.
Include nutritional information when available.

Format your response as:
🍽️ {days}-Day {diet_type.replace('_', ' ').title()} Meal Plan - €{budget:.2f} Budget

📅 Day 1:
🥣 Breakfast: [Meal name using products above] - €[cost]
🥗 Lunch: [Meal name using products above] - €[cost]  
🍽️ Dinner: [Meal name using products above] - €[cost]
🍎 Snack: [Meal name using products above] - €[cost]
📊 Nutrition: [calories] kcal, [protein]g protein

[Continue for all days...]

💰 Total estimated cost: €[total]
🥗 Average daily cost: €[average]
📊 Weekly nutrition summary

IMPORTANT: Use ONLY the products listed above. Be realistic about costs and quantities."""
        
        try:
            response = self.model.generate_content(prompt)
            yield response.text
        except Exception as e:
            yield f"Error generating meal plan: {str(e)}"


# Convenience function
async def rag_meal_plan_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Generate RAG meal plan using database retrieval."""
    planner = RAGMealPlanner()
    async for chunk in planner.generate_meal_plan(user_query, constraints):
        yield chunk
