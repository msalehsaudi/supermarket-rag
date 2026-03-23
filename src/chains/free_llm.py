"""Free LLM chains using local models or Hugging Face."""

import asyncio
import json
from typing import Dict, Any, AsyncGenerator, Optional
import requests

from src.config_free import get_llm_config, DEFAULT_FREE_LLM, FREE_LLM_TEMPERATURE, FREE_LLM_MAX_TOKENS


class FreeLLMChain:
    """
    Free LLM chain using local Ollama or Hugging Face inference.
    """
    
    def __init__(self):
        """Initialize free LLM chain."""
        self.llm_config = get_llm_config()
        self.provider = self.llm_config["provider"]
        print(f"🤖 Using LLM provider: {self.provider}")
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.llm_config.get("model", "llama2"),
                    "prompt": prompt,
                    "temperature": self.llm_config["temperature"],
                    "max_tokens": self.llm_config["max_tokens"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "Error: No response")
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    async def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face inference API."""
        try:
            from src.config_free import HUGGINGFACE_API_KEY
            if not HUGGINGFACE_API_KEY:
                return "Error: HUGGINGFACE_API_KEY not set"
            
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.llm_config["temperature"],
                    "max_new_tokens": self.llm_config["max_tokens"],
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.llm_config.get('model', 'microsoft/DialoGPT-medium')}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "Error: No text generated")
                else:
                    return str(result)
            else:
                return f"Error: Hugging Face returned status {response.status_code}"
                
        except Exception as e:
            return f"Error calling Hugging Face: {str(e)}"
    
    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate response using free LLM."""
        if self.provider == "ollama":
            response = await self._call_ollama(prompt)
            yield response
        elif self.provider == "huggingface":
            response = await self._call_huggingface(prompt)
            yield response
        else:
            yield "Error: No free LLM provider configured. Please set up Ollama or Hugging Face API key."


# Free intent classifier using simple keyword matching
class FreeIntentClassifier:
    """Simple intent classifier using keyword patterns."""
    
    def __init__(self):
        """Initialize free intent classifier."""
        pass
    
    async def classify(self, message: str) -> Dict[str, Any]:
        """Classify intent using keyword patterns."""
        message_lower = message.lower()
        
        # Simple keyword-based classification
        if any(word in message_lower for word in ["meal plan", "meal plan", "weekly", "daily meals", "menu", "diet plan"]):
            intent = "meal_plan"
        elif any(word in message_lower for word in ["budget", "cheap", "cheapest", "affordable", "save money", "shopping list"]):
            intent = "budget_basket"
        elif any(word in message_lower for word in ["calories", "protein", "nutrition", "macros", "fat", "carbs", "sugar"]):
            intent = "nutrition_query"
        elif any(word in message_lower for word in ["find", "search", "looking for", "show me", "compare"]):
            intent = "product_search"
        else:
            intent = "general"
        
        # Extract basic constraints
        constraints = {}
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', message)
        if numbers:
            if "€" in message_lower or "euro" in message_lower or "eur" in message_lower:
                constraints["budget"] = float(numbers[0])
            elif "day" in message_lower:
                constraints["days"] = int(numbers[0])
            elif "calorie" in message_lower:
                constraints["calorie_target"] = int(numbers[0])
        
        # Extract diet types
        if "vegan" in message_lower:
            constraints["diet_type"] = "vegan"
        elif "vegetarian" in message_lower:
            constraints["diet_type"] = "vegetarian"
        elif "keto" in message_lower:
            constraints["diet_type"] = "keto"
        elif "weight loss" in message_lower:
            constraints["diet_type"] = "weight_loss"
        elif "muscle" in message_lower or "protein" in message_lower:
            constraints["diet_type"] = "muscle_gain"
        
        return {
            "intent": intent,
            "constraints": constraints
        }


# Free meal planner using template responses
class FreeMealPlanner:
    """Simple meal planner using templates."""
    
    def __init__(self):
        """Initialize free meal planner."""
        pass
    
    async def generate_meal_plan(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate a simple meal plan."""
        days = constraints.get("days", 7)
        budget = constraints.get("budget", 50.0)
        diet_type = constraints.get("diet_type", "balanced")
        
        response = f"""🍽️ {days.title()}-Day {diet_type.replace('_', ' ').title()} Meal Plan - €{budget:.2f} Budget

📅 Day 1:
🥣 Breakfast: Oatmeal with berries - €2.50
🥗 Lunch: Grilled chicken salad - €8.00  
🍽️ Dinner: Pasta with vegetables - €6.00
🍎 Snack: Greek yogurt - €1.50

📅 Day 2:
🥣 Breakfast: Scrambled eggs with toast - €3.00
🥗 Lunch: Turkey sandwich - €7.50
🍽️ Dinner: Fish with rice - €9.00
🍎 Snack: Apple with peanut butter - €1.00

📅 Day 3:
🥣 Breakfast: Smoothie bowl - €3.50
🥗 Lunch: Quinoa bowl - €8.50
🍽️ Dinner: Vegetable stir-fry - €6.50
🍎 Snack: Mixed nuts - €2.00

💰 Total estimated cost: €{budget:.2f}
🥗 Average daily cost: €{budget/7:.2f}
📊 This is a template meal plan. For personalized plans, add an OpenAI API key.

⚠️  Note: This is a simplified meal plan template. 
For detailed nutritional planning and personalized recommendations, 
please configure a paid LLM API key."""
        
        yield response


# Free budget optimizer
class FreeBudgetOptimizer:
    """Simple budget optimizer using price sorting."""
    
    def __init__(self):
        """Initialize free budget optimizer."""
        pass
    
    async def optimize_budget(self, user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate budget-optimized shopping list."""
        budget = constraints.get("budget", 30.0)
        goal = constraints.get("goal", "general nutrition")
        
        # Load product data for real recommendations
        try:
            from src.ingest.loader import load_and_clean
            df = load_and_clean()
            
            # Filter and sort by price
            affordable_products = df[df['price_eur'] <= budget].sort_values('price_eur')
            
            response = f"""💰 Budget Optimization - €{budget:.2f} Budget

🎯 Goal: {goal}

📦 Recommended Products (Under Budget):
"""
            
            # Add top 10 affordable products
            for i, (_, row) in enumerate(affordable_products.head(10).iterrows()):
                response += f"{i+1}. {row['name']} ({row['brand']}) - €{row['price_eur']:.2f}\n"
                response += f"   Category: {row['category']}"
                if row['is_food']:
                    response += f" | {row['calories_per_100g']} kcal/100g"
                response += "\n"
            
            response += f"""
💡 Budget Tips:
• Buy in bulk for better value
• Look for store brands vs premium brands
• Check weekly specials
• Consider frozen alternatives

⚠️  Note: This is basic price optimization. 
For advanced budget planning with nutritional analysis, 
please configure a paid LLM API key."""
            
            yield response
            
        except Exception as e:
            yield f"Error loading product data: {str(e)}"


# Convenience functions
async def free_meal_plan_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Generate meal plan using free method."""
    planner = FreeMealPlanner()
    async for chunk in planner.generate_meal_plan(user_query, constraints):
        yield chunk


async def free_budget_optimizer_chain(user_query: str, constraints: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Optimize budget using free method."""
    optimizer = FreeBudgetOptimizer()
    async for chunk in optimizer.optimize_budget(user_query, constraints):
        yield chunk


async def free_classify_intent(message: str) -> Dict[str, Any]:
    """Classify intent using free method."""
    classifier = FreeIntentClassifier()
    return await classifier.classify(message)
