"""Query rewriter using HyDE (Hypothetical Document Embeddings)."""

import asyncio
from typing import List, Dict, Any
import openai
from openai import AsyncOpenAI

from src.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE


HYDE_PROMPT = """You are a supermarket product expert.
Given this user query: "{query}"

Write a hypothetical product description that would PERFECTLY answer this query.
Include: product name, category, nutritional info (if food), price range, and why it fits.
Be specific. 2-3 sentences max.

Hypothetical product:"""

QUERY_EXPANSION_PROMPT = """You are a supermarket search expert.
Given this user query: "{query}"

Generate 3-5 alternative search terms that would help find relevant products.
Include synonyms, related categories, and different ways to express the same need.
Return as a comma-separated list.

Alternative terms:"""

CONSTRAINT_EXTRACTION_PROMPT = """You are a supermarket query analyzer.
Extract constraints from this user query: "{query}"

Return JSON only with these keys (include only if present):
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

Constraints:"""


async def rewrite_with_hyde(query: str, llm=None) -> str:
    """
    HyDE: Generate a hypothetical document, embed that instead of the raw query.
    Dramatically improves retrieval for vague/natural queries.
    
    Args:
        query: Original user query
        llm: LLM client (optional, will create if None)
        
    Returns:
        Hypothetical product description
    """
    if llm is None:
        llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await llm.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(query=query)}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in HyDE rewriting: {e}")
        return query  # Fallback to original query


async def expand_query(query: str, llm=None) -> List[str]:
    """
    Expand query with alternative search terms.
    
    Args:
        query: Original user query
        llm: LLM client (optional, will create if None)
        
    Returns:
        List of alternative search terms
    """
    if llm is None:
        llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await llm.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=query)}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=100
        )
        
        terms_text = response.choices[0].message.content.strip()
        terms = [term.strip() for term in terms_text.split(',')]
        
        # Add original query
        return [query] + [term for term in terms if term and term != query]
        
    except Exception as e:
        print(f"Error in query expansion: {e}")
        return [query]  # Fallback to original query


async def extract_constraints(query: str, llm=None) -> Dict[str, Any]:
    """
    Extract structured constraints from natural language query.
    
    Args:
        query: Original user query
        llm: LLM client (optional, will create if None)
        
    Returns:
        Dictionary of extracted constraints
    """
    if llm is None:
        llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await llm.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "user", "content": CONSTRAINT_EXTRACTION_PROMPT.format(query=query)}
            ],
            temperature=0.0,  # Lower temperature for structured output
            max_tokens=300
        )
        
        constraints_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        constraints = json.loads(constraints_text)
        
        # Type conversion and validation
        processed_constraints = {}
        
        for key, value in constraints.items():
            if key in ['budget', 'max_price_per_item', 'min_price_per_item', 'min_protein', 'max_calories', 'max_sugar', 'min_rating']:
                processed_constraints[key] = float(value) if value is not None else None
            elif key in ['days', 'calorie_target']:
                processed_constraints[key] = int(value) if value is not None else None
            elif key in ['food_only', 'in_stock']:
                processed_constraints[key] = bool(value) if value is not None else False
            else:
                processed_constraints[key] = str(value) if value is not None else None
        
        return processed_constraints
        
    except Exception as e:
        print(f"Error in constraint extraction: {e}")
        return {}  # Fallback to empty constraints


async def rewrite_query_pipeline(query: str) -> tuple[str, List[str], Dict[str, Any]]:
    """
    Complete query rewriting pipeline.
    
    Args:
        query: Original user query
        
    Returns:
        Tuple of (hyde_query, expanded_queries, constraints)
    """
    # Run all rewriting steps in parallel
    hyde_task = rewrite_with_hyde(query)
    expansion_task = expand_query(query)
    constraints_task = extract_constraints(query)
    
    hyde_query, expanded_queries, constraints = await asyncio.gather(
        hyde_task, expansion_task, constraints_task
    )
    
    return hyde_query, expanded_queries, constraints


if __name__ == "__main__":
    # Test the query rewriter
    import asyncio
    
    async def test():
        test_queries = [
            "7-day weight loss meal plan with €60 budget",
            "high protein breakfast options under €3",
            "vegan dinner ideas this week",
            "cheapest sources of protein"
        ]
        
        for query in test_queries:
            print(f"\nOriginal: {query}")
            hyde, expanded, constraints = await rewrite_query_pipeline(query)
            print(f"HyDE: {hyde}")
            print(f"Expanded: {expanded}")
            print(f"Constraints: {constraints}")
    
    asyncio.run(test())
