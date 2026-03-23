"""Simple RAG meal planner using database without heavy dependencies."""

import asyncio
from typing import Dict, Any, AsyncGenerator, List
from langchain_core.documents import Document

from src.config import get_llm_config
from src.ingest.loader import load_and_clean
from src.ingest.doc_builder import build_documents


class SimpleRAGPlanner:
    """Simple meal planner that uses database products without complex retrieval."""
    
    def __init__(self):
        """Initialize simple RAG meal planner."""
        self.config = get_llm_config()
        self._all_documents = None
        
        # Initialize Google Gemini
        import google.generativeai as genai
        genai.configure(api_key=self.config["api_key"])
        self.model = genai.GenerativeModel(self.config["model"])
        print(f"🤖 Simple RAG Planner initialized: {self.config['model']}")
    
    async def _load_documents(self):
        """Load all food documents from database."""
        if self._all_documents is None:
            df = load_and_clean()
            all_docs = build_documents(df)
            # Filter to food products only
            self._all_documents = [doc for doc in all_docs if doc.metadata.get('is_food', False)]
            print(f"📚 Loaded {len(self._all_documents)} food products from database")
    
    def _get_relevant_products(self, constraints: Dict[str, Any], limit: int = 20) -> List[Document]:
        """Get relevant products based on simple filtering."""
        budget = constraints.get("budget", 50.0)
        diet_type = constraints.get("diet_type", "balanced")
        
        # Simple filtering based on price and basic categories
        relevant_products = []
        
        for doc in self._all_documents:
            meta = doc.metadata
            price = meta.get('price_eur', float('inf'))
            
            # Budget filter (allow items up to 10% of total budget)
            if price > budget * 0.1:
                continue
            
            # Basic diet filtering
            category = meta.get('category', '').lower()
            if diet_type == "vegan" and any(x in category for x in ['meat', 'dairy', 'fish']):
                continue
            elif diet_type == "vegetarian" and any(x in category for x in ['meat', 'fish']):
                continue
            
            relevant_products.append(doc)
            
            if len(relevant_products) >= limit:
                break
        
        # Sort by price (cheapest first)
        relevant_products.sort(key=lambda x: x.metadata.get('price_eur', float('inf')))
        
        return relevant_products
    
    def _format_products_for_prompt(self, documents: List[Document]) -> str:
        """Format retrieved products for the prompt."""
        products = []
        for doc in documents:
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
    
    async def generate_meal_plan(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate meal plan using database products + Gemini."""
        await self._load_documents()
        
        # Get relevant products from database
        products = self._get_relevant_products(constraints)
        
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

IMPORTANT: Your response must have TWO CLEAR SECTIONS:

=== 🛒 SHOPPING CART LIST ===
List ALL products needed for the entire {days}-day meal plan with exact quantities:
• Product Name (Brand) - Quantity needed - Total price
• Product Name (Brand) - Quantity needed - Total price
...
💰 Total Shopping Cost: €[total]

=== 📝 RECIPES & INSTRUCTIONS ===
For each day, provide detailed recipes using ONLY the shopping cart products:

📅 Day 1:
🥣 Breakfast: [Recipe Name]
   🛒 Ingredients: [Products from shopping cart with quantities]
   👨‍🍳 Instructions: [Step-by-step cooking instructions]
   ⏱️ Time: [Prep + cook time]
   📊 Nutrition: [Estimated nutrition]

🥗 Lunch: [Recipe Name]
   🛒 Ingredients: [Products from shopping cart with quantities]
   👨‍🍳 Instructions: [Step-by-step cooking instructions]
   ⏱️ Time: [Prep + cook time]
   📊 Nutrition: [Estimated nutrition]

🍽️ Dinner: [Recipe Name]
   🛒 Ingredients: [Products from shopping cart with quantities]
   👨‍🍳 Instructions: [Step-by-step cooking instructions]
   ⏱️ Time: [Prep + cook time]
   📊 Nutrition: [Estimated nutrition]

🍎 Snack: [Recipe Name]
   🛒 Ingredients: [Products from shopping cart with quantities]
   �‍🍳 Instructions: [Step-by-step cooking instructions]
   ⏱️ Time: [Prep + cook time]
   📊 Nutrition: [Estimated nutrition]

[Continue for all {days} days...]

📋 WEEKLY SUMMARY:
💰 Total Cost: €[total]
🥗 Total Nutrition: [Weekly totals]
🛒 Shopping Items: [Number of unique items]

CRITICAL: 
- Use ONLY products listed in AVAILABLE SUPERMARKET PRODUCTS
- Shopping cart must list exact quantities needed
- Recipes must use ONLY shopping cart ingredients
- Include real prices from product list
- Be specific about cooking instructions and times"""
        
        try:
            response = self.model.generate_content(prompt)
            yield response.text
        except Exception as e:
            yield f"Error generating meal plan: {str(e)}"


# Convenience function
async def simple_rag_meal_plan_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Generate simple RAG meal plan using database products."""
    planner = SimpleRAGPlanner()
    async for chunk in planner.generate_meal_plan(user_query, constraints):
        yield chunk
