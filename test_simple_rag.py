"""Test simple RAG meal planner with database."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.chains.simple_rag_planner import SimpleRAGPlanner

async def test_simple_rag():
    """Test simple RAG meal planner."""
    print("🧪 Testing Simple RAG Database Integration")
    print("=" * 45)
    
    try:
        planner = SimpleRAGPlanner()
        
        # Test 1: Load documents
        await planner._load_documents()
        print(f"✅ Loaded {len(planner._all_documents)} food products from database")
        
        # Test 2: Get relevant products
        constraints = {
            "days": 3,
            "budget": 30.0,
            "diet_type": "balanced"
        }
        
        products = planner._get_relevant_products(constraints)
        print(f"✅ Found {len(products)} relevant products")
        
        # Test 3: Show sample products
        print("\n📦 Sample Database Products:")
        for i, doc in enumerate(products[:5]):
            meta = doc.metadata
            print(f"  {i+1}. {meta['name']} ({meta['brand']}) - €{meta['price_eur']:.2f}")
            if meta.get('calories_per_100g'):
                print(f"     🥗 {meta['calories_per_100g']} kcal, {meta.get('protein_g_per_100g', 0)}g protein")
        
        # Test 4: Generate meal plan
        print("\n🍽️ Generating meal plan using database products...")
        query = "3-day meal plan for €30"
        
        response_parts = []
        async for chunk in planner.generate_meal_plan(query, constraints):
            response_parts.append(chunk)
        
        full_response = ''.join(response_parts)
        
        print(f"\n✅ Generated meal plan ({len(full_response)} chars)")
        print(f"📄 Preview: {full_response[:400]}...")
        
        # Check if database products are mentioned
        product_names = [doc.metadata['name'].lower() for doc in products[:10]]
        response_lower = full_response.lower()
        
        mentioned_products = []
        for product_name in product_names:
            if product_name in response_lower:
                mentioned_products.append(product_name)
        
        if mentioned_products:
            print(f"\n🎯 SUCCESS! Found {len(mentioned_products)} database products in meal plan:")
            for product in mentioned_products[:5]:
                print(f"   - {product}")
            print("\n✅ The chatbot IS using your database!")
        else:
            print("\n⚠️  No database products found in meal plan response")
            print("❌ The chatbot is NOT using your database")
        
        return len(mentioned_products) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_rag())
    if success:
        print("\n🎉 RAG Database Integration: WORKING!")
        print("📊 The chatbot uses your 10,000 product database!")
    else:
        print("\n❌ RAG Database Integration: FAILED!")
        print("🤖 The chatbot is using general knowledge, not your database")
