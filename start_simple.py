"""Simple startup that bypasses complex ingestion for immediate testing."""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Simple startup for immediate testing."""
    print("🛒 Supermarket RAG System - Simple Mode")
    print("=" * 50)
    print("🚀 Bypassing ingestion for immediate testing...")
    print()
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY", "")
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
    
    if not google_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return
    
    print(f"✅ API Configuration:")
    print(f"  🤖 LLM: Google Gemini")
    print(f"  📊 Embeddings: Available")
    print()
    
    # Test simple RAG planner directly
    try:
        from src.chains.simple_rag_planner import SimpleRAGPlanner
        
        async def test_simple():
            planner = SimpleRAGPlanner()
            constraints = {
                "days": 3,
                "budget": 30.0,
                "diet_type": "balanced"
            }
            
            print("🧪 Testing Simple RAG with Shopping Cart + Recipes")
            print("=" * 60)
            
            response_parts = []
            async for chunk in planner.generate_meal_plan("3-day meal plan for €30", constraints):
                response_parts.append(chunk)
            
            full_response = ''.join(response_parts)
            
            print("✅ Generated complete response:")
            print(full_response)
            
            # Check if shopping cart format is present
            if "🛒 SHOPPING CART LIST" in full_response:
                print("\n🎉 SUCCESS! Shopping cart + recipes format working!")
            else:
                print("\n⚠️  Shopping cart format not detected")
        
        asyncio.run(test_simple())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
