"""Intent classifier for routing user queries to appropriate chains."""

import asyncio
import json
from typing import Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE


INTENT_PROMPT = """Classify this user message into exactly one category.

Message: "{message}"

Categories:
- meal_plan: user wants a multi-day meal plan or weekly food schedule
- budget_basket: user wants cheapest products for a goal or shopping list
- nutrition_query: user asks about calories, protein, macros, or health properties
- product_search: user wants to find specific products or compare items
- general: anything else

Return JSON only: {{"intent": "...", "constraints": {{...}}}}

Constraints to extract (if present):
- budget (float, weekly total in EUR)
- days (int, number of days for meal plan)
- diet_type (string: weight_loss, muscle_gain, vegan, keto, vegetarian, balanced)
- calorie_target (int, daily kcal)
- max_price_per_item (float)
- min_price_per_item (float)
- category (string)
- brand (string)
- min_protein (float, g per 100g)
- max_calories (float, per 100g)
- max_sugar (float, g per 100g)
- min_rating (float, 1-5)
- food_only (boolean)
- in_stock (boolean)
- restrictions (list of strings: gluten_free, dairy_free, nut_free, etc.)"""


class IntentClassifier:
    """
    Classifies user intents and extracts constraints from natural language queries.
    """
    
    def __init__(self, model: str = DEFAULT_LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize the intent classifier.
        
        Args:
            model: LLM model to use
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    async def classify(self, message: str) -> Dict[str, Any]:
        """
        Classify the intent of a user message and extract constraints.
        
        Args:
            message: User message
            
        Returns:
            Dictionary with intent and constraints
        """
        if not self.client:
            # Fallback to simple keyword-based classification
            return self._fallback_classify(message)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": INTENT_PROMPT.format(message=message)}
                ],
                temperature=self.temperature,
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(content)
            
            # Validate and clean the result
            return self._validate_result(result)
            
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return self._fallback_classify(message)
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the classification result.
        
        Args:
            result: Raw result from LLM
            
        Returns:
            Validated result
        """
        # Ensure intent is valid
        valid_intents = ["meal_plan", "budget_basket", "nutrition_query", "product_search", "general"]
        intent = result.get("intent", "general")
        if intent not in valid_intents:
            intent = "general"
        
        # Clean constraints
        constraints = result.get("constraints", {})
        cleaned_constraints = {}
        
        # Type conversion and validation for constraints
        for key, value in constraints.items():
            if value is None:
                continue
                
            if key in ["budget", "max_price_per_item", "min_price_per_item", "min_protein", "max_calories", "max_sugar", "min_rating"]:
                try:
                    cleaned_constraints[key] = float(value)
                except (ValueError, TypeError):
                    continue
            elif key in ["days", "calorie_target"]:
                try:
                    cleaned_constraints[key] = int(value)
                except (ValueError, TypeError):
                    continue
            elif key in ["food_only", "in_stock"]:
                cleaned_constraints[key] = bool(value)
            elif key == "restrictions":
                if isinstance(value, list):
                    cleaned_constraints[key] = [str(item) for item in value]
                elif isinstance(value, str):
                    cleaned_constraints[key] = [value]
            else:
                cleaned_constraints[key] = str(value)
        
        return {
            "intent": intent,
            "constraints": cleaned_constraints
        }
    
    def _fallback_classify(self, message: str) -> Dict[str, Any]:
        """
        Fallback classification using keyword matching.
        
        Args:
            message: User message
            
        Returns:
            Basic classification result
        """
        message_lower = message.lower()
        
        # Simple keyword-based intent detection
        if any(word in message_lower for word in ["meal plan", "meal plan", "weekly", "daily meals", "menu"]):
            intent = "meal_plan"
        elif any(word in message_lower for word in ["budget", "cheap", "cheapest", "affordable", "save money"]):
            intent = "budget_basket"
        elif any(word in message_lower for word in ["calories", "protein", "nutrition", "macros", "fat", "carbs", "sugar"]):
            intent = "nutrition_query"
        elif any(word in message_lower for word in ["find", "search", "looking for", "show me", "compare"]):
            intent = "product_search"
        else:
            intent = "general"
        
        # Extract basic constraints
        constraints = {}
        
        # Budget extraction
        if "€" in message or "eur" in message_lower:
            import re
            budget_match = re.search(r'[€\s]?(\d+(?:\.\d+)?)\s*(?:€|eur)', message_lower)
            if budget_match:
                try:
                    constraints["budget"] = float(budget_match.group(1))
                except ValueError:
                    pass
        
        return {
            "intent": intent,
            "constraints": constraints
        }


# Convenience function
async def classify_intent(message: str) -> Dict[str, Any]:
    """
    Classify the intent of a user message.
    
    Args:
        message: User message
        
    Returns:
        Dictionary with intent and constraints
    """
    classifier = IntentClassifier()
    return await classifier.classify(message)


if __name__ == "__main__":
    # Test the intent classifier
    async def test():
        classifier = IntentClassifier()
        
        test_messages = [
            "Give me a 7-day weight loss meal plan with €60 budget",
            "What are the highest protein foods under €3?",
            "How many calories are in chicken breast?",
            "Find me organic vegetables from Spain",
            "What's the weather like?"
        ]
        
        for message in test_messages:
            print(f"\nMessage: {message}")
            result = await classifier.classify(message)
            print(f"Intent: {result['intent']}")
            print(f"Constraints: {result['constraints']}")
    
    asyncio.run(test())
