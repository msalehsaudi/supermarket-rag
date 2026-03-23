"""Working startup that bypasses complex ingestion issues."""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Working startup for immediate use."""
    print("🛒 Supermarket RAG System - Working Mode")
    print("=" * 50)
    print("🚀 Bypassing ingestion - using direct database access")
    print()
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY", "")
    
    if not google_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return
    
    print(f"✅ API Configuration:")
    print(f"  🤖 LLM: Google Gemini")
    print(f"  📊 Database: Direct access (10,000 products)")
    print()
    
    # Test simple RAG planner directly
    try:
        from src.chains.simple_rag_planner import SimpleRAGPlanner
        
        async def test_working():
            planner = SimpleRAGPlanner()
            constraints = {
                "days": 3,
                "budget": 30.0,
                "diet_type": "balanced"
            }
            
            print("🧪 Testing Working RAG System")
            print("=" * 60)
            
            response_parts = []
            async for chunk in planner.generate_meal_plan("3-day meal plan for €30", constraints):
                response_parts.append(chunk)
            
            full_response = ''.join(response_parts)
            
            print("✅ Generated complete response:")
            print(full_response[:600] + "..." if len(full_response) > 600 else full_response)
            
            # Check if shopping cart format is present
            if "🛒 SHOPPING CART LIST" in full_response:
                print("\n🎉 SUCCESS! Shopping cart + recipes working!")
            else:
                print("\n⚠️  Format issue detected")
        
        asyncio.run(test_working())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
