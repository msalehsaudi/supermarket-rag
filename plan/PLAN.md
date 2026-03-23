# 🛒 Supermarket RAG Chatbot — Project Plan

> **Stack:** ChromaDB · LangChain · OpenAI Embeddings · Gemini 2.5 Pro · FastAPI · React  
> **Data:** `D:\mo-labs\rag-project\data\supermarket_dataset.csv` (10,000 products)  
> **Goal:** Conversational AI that generates meal plans, budget baskets, and nutrition advice from real supermarket data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack & Rationale](#3-tech-stack--rationale)
4. [Project Structure](#4-project-structure)
5. [Phase 1 — Data Ingestion](#5-phase-1--data-ingestion)
6. [Phase 2 — Vector Store Setup](#6-phase-2--vector-store-setup)
7. [Phase 3 — Retrieval Pipeline](#7-phase-3--retrieval-pipeline)
8. [Phase 4 — LLM Chains](#8-phase-4--llm-chains)
9. [Phase 5 — API Layer](#9-phase-5--api-layer)
10. [Phase 6 — Frontend UI](#10-phase-6--frontend-ui)
11. [Environment Setup](#11-environment-setup)
12. [Cline Rules & Skills](#12-cline-rules--skills)
13. [Testing Strategy](#13-testing-strategy)
14. [Deployment](#14-deployment)
15. [Roadmap](#15-roadmap)

---

## 1. Project Overview

A production-grade RAG chatbot that answers natural language queries about supermarket products. Users can ask for:

- **7-day meal plans** tailored to diet goals (weight loss, muscle gain, vegan, keto, etc.)
- **Budget-optimized shopping lists** (e.g. "full week of meals under €50")
- **Nutrition lookups** (e.g. "highest protein options under €3/100g")
- **Product discovery** (e.g. "Italian cheeses with >4 star rating")
- **Smart substitutions** (e.g. "alternatives to chicken breast for protein")

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `text-embedding-3-small` | Best cost/quality, 1536-dim |
| Vector DB | ChromaDB (local) | Zero infra, persistent, metadata-filterable |
| Retrieval | Hybrid vector + BM25 | Handles both semantic and exact queries |
| LLM | Gemini 2.5 Pro | Already in Cline, excellent JSON output |
| Context strategy | HyDE + query expansion | Improves vague/natural queries dramatically |
| Output format | Structured JSON → markdown | Parseable by frontend, readable by users |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  CSV → Doc Builder → Chunker → Metadata Store                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  EMBED LAYER                                                    │
│  text-embedding-3-small → Batch Embedder (100/req) → Cache     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  VECTOR STORE                                                   │
│  ChromaDB (cosine similarity) + Metadata Filters               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  RETRIEVAL LAYER                                                │
│  Query Rewriter (HyDE) → Hybrid Retriever → Reranker → Top-K  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  GENERATION LAYER                                               │
│  Intent Classifier → Chain Router → Prompt Builder → Gemini    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  API LAYER                                                      │
│  FastAPI → Streaming SSE → React Frontend                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data flow for a meal plan query

```
User: "7-day weight loss meal plan, €60 budget"
  │
  ▼
Intent classifier → { type: "meal_plan", budget: 60, goal: "weight_loss" }
  │
  ▼
HyDE rewriter → Hypothetical product description for each meal slot
  │
  ▼
ChromaDB query → is_food=true, price_eur<=8, in_stock=true → top 40 candidates
  │
  ▼
BM25 + vector fusion → top 20 after RRF scoring
  │
  ▼
Cross-encoder reranker → top 10 per meal slot
  │
  ▼
Meal plan chain → 7 days × 4 meals = JSON plan with macros + cost
  │
  ▼
Streaming response → frontend renders day-by-day
```

---

## 3. Tech Stack & Rationale

### Backend

| Package | Version | Purpose |
|---|---|---|
| `langchain` | ≥0.2.0 | RAG chain orchestration |
| `langchain-community` | ≥0.2.0 | BM25, retrievers |
| `langchain-openai` | ≥0.1.0 | OpenAI embeddings integration |
| `chromadb` | ≥0.5.0 | Vector store (local persistent) |
| `openai` | ≥1.30.0 | Embeddings API |
| `fastapi` | ≥0.111.0 | Async REST + SSE streaming |
| `uvicorn` | ≥0.30.0 | ASGI server |
| `pandas` | ≥2.0.0 | CSV ingestion and processing |
| `rank-bm25` | ≥0.2.2 | Keyword-based retrieval |
| `sentence-transformers` | ≥3.0.0 | Cross-encoder reranking |
| `python-dotenv` | ≥1.0.0 | Environment variable management |
| `pydantic` | ≥2.0.0 | Request/response validation |
| `pytest` | ≥8.0.0 | Testing framework |
| `pytest-asyncio` | ≥0.23.0 | Async test support |

### Frontend

| Package | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool |
| Tailwind CSS | Styling |
| Framer Motion | Animations |
| React Markdown | Render LLM markdown output |

### LLM Recommendation for Cline

Use `kimi-k2-instruct-0905` (256K context) as primary, `deepseek-v3.2` as fallback. Both are free endpoints on OpenRouter.

---

## 4. Project Structure

```
rag-project/
├── data/
│   └── supermarket_dataset.csv          # Source data (10k products)
│
├── src/
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── loader.py                    # CSV → pandas DataFrame
│   │   ├── doc_builder.py               # Row → rich Document text
│   │   └── embedder.py                  # Batch embed + checkpoint
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chroma_store.py              # ChromaDB client + collection
│   │   └── schema.py                    # Metadata field definitions
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_rewriter.py            # HyDE + query expansion
│   │   ├── hybrid_retriever.py          # Vector + BM25 fusion (RRF)
│   │   └── reranker.py                  # Cross-encoder / MMR
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py         # Route query to correct chain
│   │   ├── meal_planner.py              # 7-day meal plan chain
│   │   ├── budget_optimizer.py          # Cheapest basket chain
│   │   ├── nutrition_advisor.py         # Macro/diet-aware chain
│   │   └── product_search.py            # General product discovery
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app + routes
│   │   ├── models.py                    # Pydantic request/response models
│   │   └── streaming.py                 # SSE streaming helpers
│   │
│   └── config.py                        # Central config (env vars, constants)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── MealPlanCard.jsx
│   │   │   ├── ProductCard.jsx
│   │   │   └── StreamingText.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_chains.py
│   └── test_api.py
│
├── chroma_db/                           # Auto-created by ChromaDB
│   └── .ingest_checkpoint.json          # Tracks embedded products
│
├── .clinerules                          # Cline behavior rules
├── .clineskills/
│   └── rag_skill.md                     # Cline skill definition
├── .env.example                         # Environment variable template
├── requirements.txt
├── README.md
└── PLAN.md                              # This file
```

---

## 5. Phase 1 — Data Ingestion

**Goal:** Transform the raw CSV into rich, embeddable text documents with structured metadata.

### 5.1 Document design

The most critical decision in RAG quality. Each product becomes a natural-language sentence that embeds well semantically.

**File:** `src/ingest/doc_builder.py`

```python
def build_document_text(row: pd.Series) -> str:
    """
    Convert a product row to rich retrievable text.
    This text is what gets embedded — quality here = quality of retrieval.
    """
    parts = [
        f"{row['name']} is a {row['category'].lower()} product",
        f"made by {row['brand']} from {row['origin']}.",
        f"It weighs {row['weight_kg']}kg and costs €{row['price_eur']}.",
    ]

    if row['is_food'] == 'Yes':
        parts.append(
            f"Per 100g it contains {row['calories_per_100g']} calories, "
            f"{row['protein_g_per_100g']}g protein, "
            f"{row['fat_g_per_100g']}g fat, "
            f"{row['sugar_g_per_100g']}g sugar, "
            f"{row['carbs_g_per_100g']}g carbs, "
            f"{row['fiber_g_per_100g']}g fiber."
        )

    stock = "currently in stock" if row['in_stock'] == 'Yes' else "currently out of stock"
    parts.append(f"This product is {stock} with a {row['rating']}/5 rating.")
    
    return " ".join(parts)
```

### 5.2 Metadata schema

**File:** `src/vectorstore/schema.py`

```python
METADATA_FIELDS = {
    "product_id":           int,
    "sku":                  str,
    "name":                 str,
    "brand":                str,
    "category":             str,
    "origin":               str,
    "weight_kg":            float,
    "price_eur":            float,
    "in_stock":             bool,   # True/False (not "Yes"/"No")
    "rating":               float,
    "is_food":              bool,
    "calories_per_100g":    float,  # 0.0 for non-food
    "protein_g_per_100g":   float,
    "fat_g_per_100g":       float,
    "sugar_g_per_100g":     float,
    "carbs_g_per_100g":     float,
    "fiber_g_per_100g":     float,
    "sodium_mg_per_100g":   float,
}
```

### 5.3 Batch embedder with checkpoint

**File:** `src/ingest/embedder.py`

```python
import json, os
from pathlib import Path
from openai import AsyncOpenAI

CHECKPOINT = Path("./chroma_db/.ingest_checkpoint.json")
BATCH_SIZE = 100

async def embed_and_upsert(documents, collection):
    checkpoint = load_checkpoint()
    client = AsyncOpenAI()
    
    pending = [d for d in documents if d.metadata["product_id"] not in checkpoint]
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    
    for i, batch in enumerate(batches):
        texts = [d.page_content for d in batch]
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        vectors = [r.embedding for r in response.data]
        
        collection.upsert(
            ids=[str(d.metadata["product_id"]) for d in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[d.metadata for d in batch]
        )
        
        for d in batch:
            checkpoint.add(d.metadata["product_id"])
        save_checkpoint(checkpoint)
        print(f"Batch {i+1}/{len(batches)} done — {len(checkpoint)} products embedded")
```

### 5.4 Run ingestion

```bash
python -m src.ingest.embedder
# Expected: ~10 minutes, ~$0.01 at text-embedding-3-small pricing
# Output: ./chroma_db/ populated with 10,000 vectors
```

---

## 6. Phase 2 — Vector Store Setup

**File:** `src/vectorstore/chroma_store.py`

```python
import chromadb
from chromadb.utils import embedding_functions

def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    
    return client.get_or_create_collection(
        name="supermarket_products",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

def build_metadata_filter(constraints: dict) -> dict:
    """
    Build a ChromaDB $where filter from extracted constraints.
    
    Example input:
        { "max_price": 5.0, "min_protein": 15.0, "food_only": True, "in_stock": True }
    
    Example output:
        { "$and": [
            {"is_food": {"$eq": True}},
            {"price_eur": {"$lte": 5.0}},
            {"protein_g_per_100g": {"$gte": 15.0}},
            {"in_stock": {"$eq": True}}
        ]}
    """
    filters = []
    
    if constraints.get("food_only"):
        filters.append({"is_food": {"$eq": True}})
    if constraints.get("in_stock"):
        filters.append({"in_stock": {"$eq": True}})
    if "max_price" in constraints:
        filters.append({"price_eur": {"$lte": constraints["max_price"]}})
    if "min_price" in constraints:
        filters.append({"price_eur": {"$gte": constraints["min_price"]}})
    if "min_protein" in constraints:
        filters.append({"protein_g_per_100g": {"$gte": constraints["min_protein"]}})
    if "max_calories" in constraints:
        filters.append({"calories_per_100g": {"$lte": constraints["max_calories"]}})
    if "category" in constraints:
        filters.append({"category": {"$eq": constraints["category"]}})
    if "min_rating" in constraints:
        filters.append({"rating": {"$gte": constraints["min_rating"]}})
    
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters} if filters else {}
```

---

## 7. Phase 3 — Retrieval Pipeline

### 7.1 HyDE query rewriter

**File:** `src/retrieval/query_rewriter.py`

```python
HYDE_PROMPT = """You are a supermarket product expert.
Given this user query: "{query}"

Write a hypothetical product description that would PERFECTLY answer this query.
Include: product name, category, nutritional info (if food), price range, and why it fits.
Be specific. 2-3 sentences max.

Hypothetical product:"""

async def rewrite_with_hyde(query: str, llm) -> str:
    """
    HyDE: Generate a hypothetical document, embed that instead of the raw query.
    Dramatically improves retrieval for vague/natural queries.
    """
    response = await llm.ainvoke(HYDE_PROMPT.format(query=query))
    return response.content
```

### 7.2 Hybrid retriever

**File:** `src/retrieval/hybrid_retriever.py`

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

def build_hybrid_retriever(chroma_collection, all_documents, k: int = 20):
    """
    Combine vector similarity (semantic) with BM25 (keyword).
    Weights: 60% vector, 40% BM25 — tune based on query type.
    """
    # Vector retriever (semantic search)
    chroma_retriever = Chroma(
        client=chroma_collection._client,
        collection_name="supermarket_products",
    ).as_retriever(search_kwargs={"k": k})
    
    # BM25 retriever (exact keyword match)
    bm25_retriever = BM25Retriever.from_documents(all_documents, k=k)
    
    # Ensemble with Reciprocal Rank Fusion
    return EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
```

### 7.3 Reranker

**File:** `src/retrieval/reranker.py`

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list, k: int = 5) -> list:
    """
    Cross-encoder reranking — more accurate than bi-encoder cosine similarity.
    Use AFTER hybrid retrieval to narrow k=20 → k=5.
    """
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in scored[:k]]
```

---

## 8. Phase 4 — LLM Chains

### 8.1 Intent classifier

**File:** `src/chains/intent_classifier.py`

```python
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
- category (string)
- restrictions (list of strings: gluten_free, dairy_free, nut_free, etc.)"""
```

### 8.2 Meal planner chain

**File:** `src/chains/meal_planner.py`

```python
MEAL_PLAN_PROMPT = """You are a professional nutritionist and budget-conscious meal planner.

USER REQUEST: {user_query}

CONSTRAINTS:
- Diet goal: {diet_type}
- Daily calorie target: {calorie_target} kcal  
- Weekly budget: €{budget}
- Dietary restrictions: {restrictions}
- Days: {days}

AVAILABLE SUPERMARKET PRODUCTS (retrieved from database):
{retrieved_products}

Create a {days}-day meal plan using ONLY products from the list above.
For each day, provide breakfast, lunch, dinner, and a snack.

Return as JSON:
{{
  "summary": {{
    "total_cost": float,
    "avg_daily_calories": float,
    "avg_daily_protein": float
  }},
  "days": [
    {{
      "day": 1,
      "meals": {{
        "breakfast": {{
          "products": [{{ "name": str, "quantity": str, "cost": float, "calories": float, "protein": float }}],
          "total_calories": float,
          "total_cost": float
        }},
        "lunch": {{ ... }},
        "dinner": {{ ... }},
        "snack": {{ ... }}
      }},
      "day_total_cost": float,
      "day_total_calories": float
    }}
  ]
}}"""
```

### 8.3 Budget optimizer chain

**File:** `src/chains/budget_optimizer.py`

Goal: Find the cheapest combination of products that meets a nutritional target.

```python
BUDGET_PROMPT = """You are a budget-conscious shopping advisor.

USER REQUEST: {user_query}

BUDGET: €{budget}
GOAL: {goal}

AVAILABLE PRODUCTS (sorted by price/nutritional value):
{retrieved_products}

Build the most cost-effective shopping basket.
- Prioritize products with best value per euro for the stated goal
- Do not exceed the budget
- Prefer in-stock items
- Group by category

Return JSON with: items[], total_cost, total_macros, savings_tips[]"""
```

---

## 9. Phase 5 — API Layer

**File:** `src/api/main.py`

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Supermarket RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint — routes to the appropriate chain."""
    intent_data = await classify_intent(request.message)
    
    chain_map = {
        "meal_plan":       meal_planner_chain,
        "budget_basket":   budget_optimizer_chain,
        "nutrition_query": nutrition_advisor_chain,
        "product_search":  product_search_chain,
        "general":         general_rag_chain,
    }
    
    chain = chain_map.get(intent_data["intent"], general_rag_chain)
    
    return StreamingResponse(
        chain(request.message, intent_data["constraints"]),
        media_type="text/event-stream"
    )

@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks):
    """Trigger re-ingestion of the CSV (idempotent via checkpoint)."""
    background_tasks.add_task(run_ingestion_pipeline)
    return {"status": "started", "message": "Ingestion running in background"}

@app.get("/stats")
async def stats():
    """Return collection stats for the admin panel."""
    collection = get_collection()
    return {
        "total_products": collection.count(),
        "chroma_path": "./chroma_db",
        "embedding_model": "text-embedding-3-small",
    }
```

### API request/response models

**File:** `src/api/models.py`

```python
from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    history: Optional[List[dict]] = []   # [{"role": "user/assistant", "content": str}]

class ChatResponse(BaseModel):
    response: str
    intent: str
    sources: List[dict]                  # Products used in the answer
    metadata: dict                       # Cost, macros, etc.
```

---

## 10. Phase 6 — Frontend UI

### Component overview

```
App
├── Sidebar
│   ├── Logo
│   ├── ConversationHistory
│   └── NewChatButton
│
└── MainPanel
    ├── ChatHeader (title + stats)
    ├── ChatWindow
    │   ├── WelcomeScreen (empty state)
    │   └── MessageList
    │       ├── UserMessage
    │       └── AssistantMessage
    │           ├── StreamingText
    │           ├── MealPlanCard (structured render)
    │           ├── ProductGrid (product cards)
    │           └── NutritionTable
    └── InputArea
        ├── TextInput
        ├── SuggestionChips
        └── SendButton
```

### Suggestion chips (pre-built queries)

```js
const SUGGESTIONS = [
  "7-day weight loss meal plan, €60 budget",
  "Highest protein foods under €3",
  "Vegan dinner ideas this week",
  "Cheapest breakfast options",
  "Muscle gain shopping list, €80",
  "Low sugar snacks under 100 calories",
];
```

### Streaming implementation

```js
async function sendMessage(message) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history: conversationHistory }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    setStreamingText(prev => prev + chunk);
  }
}
```

---

## 11. Environment Setup

### `.env.example`

```bash
# OpenAI — for embeddings (text-embedding-3-small)
OPENAI_API_KEY=sk-...

# Google — for Gemini 2.5 Pro (or use via OpenRouter)
GOOGLE_API_KEY=...

# OpenRouter — alternative LLM access (kimi-k2, deepseek-v3.2, etc.)
OPENROUTER_API_KEY=sk-or-...

# App settings
CHROMA_DB_PATH=./chroma_db
CSV_DATA_PATH=D:/mo-labs/rag-project/data/supermarket_dataset.csv
API_PORT=8000
EMBEDDING_BATCH_SIZE=100
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
```

### First-time setup

```bash
# 1. Clone / navigate to project
cd D:\mo-labs\rag-project

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill env vars
cp .env.example .env
# Edit .env with your API keys

# 5. Run ingestion (one-time, ~10 min)
python -m src.ingest.embedder

# 6. Verify vector store
python -c "
import chromadb
c = chromadb.PersistentClient('./chroma_db')
col = c.get_collection('supermarket_products')
print(f'Products embedded: {col.count()}')
"

# 7. Start API
uvicorn src.api.main:app --reload --port 8000

# 8. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## 12. Cline Rules & Skills

### `.clinerules`

```markdown
# Cline Rules — Supermarket RAG Project

## Context
RAG chatbot over 10k supermarket products.
Vector DB: ChromaDB at ./chroma_db
Embeddings: OpenAI text-embedding-3-small (1536-dim)
LLM: Gemini 2.5 Pro / kimi-k2-instruct-0905
Framework: LangChain + FastAPI + React

## Code conventions
- All Python is async — use async/await everywhere, no blocking I/O
- Type hints required on ALL function signatures
- Pydantic models for all request/response schemas
- Environment variables via python-dotenv — NEVER hardcode API keys
- Batch all embedding calls — max 100 items per OpenAI request
- All retrieval returns List[Document] with metadata intact
- Use upsert (not add) for ChromaDB — supports re-ingestion

## RAG rules
- ALWAYS use metadata pre-filtering before vector search for numerical constraints
- ALWAYS apply HyDE rewriting before embedding the user query
- Use k=20 for initial retrieval, rerank to k=5 before passing to LLM
- Food product documents MUST include full nutritional text
- Exclude non-food items (is_food=False) from meal plan chains
- Never invent product data — only use what's retrieved from ChromaDB

## Available metadata filters (ChromaDB $where)
- price_eur: float — $lte, $gte
- calories_per_100g: float — $lte, $gte
- protein_g_per_100g: float — $lte, $gte
- fat_g_per_100g: float — $lte, $gte
- carbs_g_per_100g: float — $lte, $gte
- category: str — $eq
- is_food: bool — $eq
- in_stock: bool — $eq
- brand: str — $eq
- rating: float — $gte
- origin: str — $eq

## File ownership
- New retrieval strategies → src/retrieval/
- New chains → src/chains/
- No business logic in src/api/main.py — routing only
- Shared constants → src/config.py

## Testing
- Every new chain needs tests in tests/ with ≥3 example queries
- Always test: meal plan query, budget query, nutrition query
- Run pytest tests/ -v before marking any task done
```

### `.clineskills/rag_skill.md`

```markdown
# Skill: Add a new RAG chain

## When to use
User wants a new query type or capability added to the chatbot.

## Steps
1. Add intent branch in src/chains/intent_classifier.py
2. Add route in src/api/main.py chain_map
3. Create src/chains/my_chain.py using this skeleton:

async def my_chain(query: str, constraints: dict) -> AsyncGenerator[str, None]:
    # 1. Rewrite query
    rewritten = await rewrite_with_hyde(query, llm)
    
    # 2. Build metadata filter from constraints
    where_filter = build_metadata_filter(constraints)
    
    # 3. Hybrid retrieval with filter
    docs = await hybrid_retriever.aget_relevant_documents(
        rewritten, where=where_filter
    )
    
    # 4. Rerank
    reranked = reranker.rerank(query, docs, k=5)
    
    # 5. Build context string
    context = format_products_for_prompt(reranked)
    
    # 6. Stream LLM response
    prompt = MY_PROMPT_TEMPLATE.format(
        user_query=query,
        retrieved_products=context,
        **constraints
    )
    async for chunk in llm.astream(prompt):
        yield chunk.content

4. Add frontend handling in ChatWindow.jsx for the new response type
```

---

## 13. Testing Strategy

### Unit tests

```bash
# Test ingestion
pytest tests/test_ingestion.py -v
# - CSV loads correctly (10k rows)
# - Document text includes nutritional info for food items
# - Metadata types are correct (bool, float, not strings)

# Test retrieval
pytest tests/test_retrieval.py -v
# - ChromaDB returns results for semantic queries
# - Metadata filters work (price_eur <= 5.0)
# - HyDE improves retrieval vs raw query (measure recall@5)

# Test chains
pytest tests/test_chains.py -v
# - Meal plan chain returns valid JSON
# - Plan stays within budget constraint
# - Budget optimizer finds cheapest valid basket
```

### Integration tests (example queries)

```python
TEST_QUERIES = [
    # Meal planning
    "Give me a 7-day weight loss meal plan, budget €60",
    "Create a vegan meal plan for the week",
    "High protein meal plan for muscle gain, €80 budget",
    
    # Budget
    "Cheapest breakfast options under €1 per serving",
    "Build me a €30 shopping list for a week",
    
    # Nutrition
    "Which products have the most protein per euro?",
    "Low calorie snacks under 50 calories per 100g",
    "Best sources of fiber in the store",
    
    # Product search
    "Italian cheeses with rating above 4",
    "Organic vegetables from Spain",
]
```

---

## 14. Deployment

### Local (development)

```bash
# Backend
uvicorn src.api.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
```

### Docker (production)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./chroma_db:/app/chroma_db
  
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [api]
```

---

## 15. Roadmap

### v1.0 — MVP (current plan)
- [x] Data ingestion + ChromaDB
- [x] Hybrid retrieval (vector + BM25)
- [x] HyDE query rewriting
- [x] Meal plan chain
- [x] Budget optimizer chain
- [x] FastAPI with SSE streaming
- [x] React frontend with suggestion chips

### v1.1 — Enhanced retrieval
- [ ] Multi-query retrieval (generate 3 variations, merge results)
- [ ] Contextual compression (remove irrelevant sentences from retrieved docs)
- [ ] Add product images (scrape or generate)
- [ ] Conversation memory (LangChain ConversationBufferMemory)

### v1.2 — Advanced features
- [ ] Dietary profile saving (user preferences persist across sessions)
- [ ] Nutritional tracking (add meals to a daily log)
- [ ] Price comparison over time (track price changes)
- [ ] Export meal plan as PDF

### v2.0 — Production
- [ ] Switch to Qdrant (production vector DB with better performance)
- [ ] Add evaluation pipeline (RAGAS metrics: faithfulness, answer relevance)
- [ ] A/B test retrieval strategies
- [ ] Real supermarket API integration (replace CSV with live data)

---

*Last updated: March 2026*  
*Maintained by: mo-labs*
