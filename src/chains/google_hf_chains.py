"""LLM chains using Google Gemini and Hugging Face."""

import asyncio
import google.generativeai as genai
from typing import Dict, Any, AsyncGenerator, Optional

from src.config import get_llm_config, GOOGLE_API_KEY


class GoogleGeminiChain:
    """
    LLM chain using Google Gemini API.
    """
    
    def __init__(self):
        """Initialize Google Gemini chain."""
        self.config = get_llm_config()
        
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.config["model"])
        print(f"🤖 Google Gemini initialized: {self.config['model']}")
    
    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate response using Google Gemini."""
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response.text:
                yield response.text
            else:
                yield "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            yield f"Error with Gemini API: {str(e)}"


# Intent classifier using Gemini
class GoogleIntentClassifier:
    """Intent classifier using Google Gemini."""
    
    def __init__(self):
        """Initialize intent classifier."""
        self.chain = GoogleGeminiChain()
    
    async def classify(self, message: str) -> Dict[str, Any]:
        """Classify intent using Gemini."""
        prompt = f"""Classify the user's intent and extract constraints from this message:

User message: "{message}"

Please respond in this JSON format:
{{
    "intent": "meal_plan|budget_basket|nutrition_query|product_search|general",
    "constraints": {{
        "budget": number or null,
        "days": number or null,
        "diet_type": "vegan|vegetarian|keto|weight_loss|muscle_gain|balanced" or null,
        "calorie_target": number or null,
        "keywords": ["keyword1", "keyword2"] or null
    }}
}}

Only respond with valid JSON, no other text."""

        try:
            response_text = ""
            async for chunk in self.chain.generate_response(prompt):
                response_text += chunk
            
            # Parse JSON response
            import json
            result = json.loads(response_text)
            
            # Validate and clean result
            if "intent" not in result:
                result["intent"] = "general"
            if "constraints" not in result:
                result["constraints"] = {}
            
            return result
            
        except Exception as e:
            print(f"Error classifying intent: {e}")
            # Fallback to simple classification
            return self._fallback_classify(message)
    
    def _fallback_classify(self, message: str) -> Dict[str, Any]:
        """Simple fallback classification."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["meal plan", "weekly", "daily meals", "menu"]):
            intent = "meal_plan"
        elif any(word in message_lower for word in ["budget", "cheap", "cheapest", "affordable"]):
            intent = "budget_basket"
        elif any(word in message_lower for word in ["calories", "protein", "nutrition", "macros"]):
            intent = "nutrition_query"
        elif any(word in message_lower for word in ["find", "search", "looking for", "show me"]):
            intent = "product_search"
        else:
            intent = "general"
        
        return {"intent": intent, "constraints": {}}


# Meal planner using Gemini
class GoogleMealPlanner:
    """Meal planner using Google Gemini."""
    
    def __init__(self):
        """Initialize meal planner."""
        self.chain = GoogleGeminiChain()
    
    async def generate_meal_plan(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate meal plan using Gemini."""
        days = constraints.get("days", 7)
        budget = constraints.get("budget", 50.0)
        diet_type = constraints.get("diet_type", "balanced")
        
        prompt = f"""Create a detailed {days}-day meal plan for a {diet_type} diet with a €{budget:.2f} budget.

User request: "{user_query}"

Requirements:
- Create specific, realistic meals
- Include breakfast, lunch, dinner, and snacks
- Estimate costs for each meal
- Ensure total cost stays within budget
- Include nutritional information (calories, protein)
- Make it practical and easy to prepare

Format your response as:
🍽️ {days}-Day {diet_type.replace('_', ' ').title()} Meal Plan - €{budget:.2f} Budget

📅 Day 1:
🥣 Breakfast: [Meal name] - €[cost]
🥗 Lunch: [Meal name] - €[cost]  
🍽️ Dinner: [Meal name] - €[cost]
🍎 Snack: [Meal name] - €[cost]
📊 Nutrition: [calories] kcal, [protein]g protein

[Continue for all days...]

💰 Total estimated cost: €[total]
🥗 Average daily cost: €[average]
📊 Weekly nutrition summary

Make sure the meal plan is realistic and within budget."""

        async for chunk in self.chain.generate_response(prompt):
            yield chunk


# Budget optimizer using Gemini
class GoogleBudgetOptimizer:
    """Budget optimizer using Google Gemini."""
    
    def __init__(self):
        """Initialize budget optimizer."""
        self.chain = GoogleGeminiChain()
    
    async def optimize_budget(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Optimize budget using Gemini."""
        budget = constraints.get("budget", 30.0)
        goal = constraints.get("goal", "general nutrition")
        
        prompt = f"""Create an optimized shopping list for a €{budget:.2f} budget focused on {goal}.

User request: "{user_query}"

Requirements:
- Focus on high nutritional value per euro
- Include variety of food categories
- Prioritize whole foods over processed
- Consider bulk purchases for better value
- Include estimated costs and quantities
- Make it practical for a week

Format your response as:
💰 Budget Optimization - €{budget:.2f} Budget
🎯 Goal: {goal}

📦 Recommended Shopping List:

🥬 Vegetables:
- [Item] - €[price] ([quantity])
- [Item] - €[price] ([quantity])

🥩 Proteins:
- [Item] - €[price] ([quantity])
- [Item] - €[price] ([quantity])

🌾 Grains/Starches:
- [Item] - €[price] ([quantity])
- [Item] - €[price] ([quantity])

🥛 Dairy/Alternatives:
- [Item] - €[price] ([quantity])
- [Item] - €[price] ([quantity])

💡 Budget Tips:
• [Tip 1]
• [Tip 2]
• [Tip 3]

💰 Total estimated cost: €[total]
💡 Savings tips for staying within budget"""

        async for chunk in self.chain.generate_response(prompt):
            yield chunk


# Product search using Gemini
class GoogleProductSearch:
    """Product search using Gemini."""
    
    def __init__(self):
        """Initialize product search."""
        self.chain = GoogleGeminiChain()
    
    async def search_products(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Search products using Gemini."""
        prompt = f"""Help the user find products based on their query.

User request: "{user_query}"

Requirements:
- Search through the supermarket database
- Find relevant products matching their criteria
- Compare prices, brands, and features
- Provide specific product recommendations
- Include nutritional information when relevant

Format your response as:
🔍 Product Search Results

🏆 Top Recommendations:

1. [Product Name] ([Brand]) - €[price]
   📊 Category: [Category]
   ⭐ Rating: [Rating]/5
   🥗 Nutrition: [Calories] kcal, [Protein]g protein per 100g
   📍 Origin: [Origin]
   💡 Why it's a good choice: [Reason]

2. [Product Name] ([Brand]) - €[price]
   [Similar format...]

📋 Comparison Summary:
• Best value: [Product]
• Highest rated: [Product]
• Most nutritious: [Product]

💡 Shopping Tips:
• [Tip 1]
• [Tip 2]

Focus on providing helpful, actionable product recommendations."""

        async for chunk in self.chain.generate_response(prompt):
            yield chunk


# Convenience functions
async def google_classify_intent(message: str) -> Dict[str, Any]:
    """Classify intent using Google Gemini."""
    classifier = GoogleIntentClassifier()
    return await classifier.classify(message)


async def google_meal_plan_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Generate meal plan using Google Gemini."""
    planner = GoogleMealPlanner()
    async for chunk in planner.generate_meal_plan(user_query, constraints):
        yield chunk


async def google_budget_optimizer_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Optimize budget using Google Gemini."""
    optimizer = GoogleBudgetOptimizer()
    async for chunk in optimizer.optimize_budget(user_query, constraints):
        yield chunk


async def google_product_search_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Search products using Google Gemini."""
    searcher = GoogleProductSearch()
    async for chunk in searcher.search_products(user_query, constraints):
        yield chunk
