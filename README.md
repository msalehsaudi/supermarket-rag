# 🛒 Supermarket RAG Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot that answers natural language queries about supermarket products. Users can ask for meal plans, budget optimization, nutrition advice, and product discovery.

## 🚀 Features

- **🍽️ Meal Planning**: Generate personalized meal plans based on dietary goals and budget
- **💰 Budget Optimization**: Find the most cost-effective shopping baskets
- **🥗 Nutrition Advice**: Get expert nutrition advice using real product data
- **🔍 Product Search**: Find and compare specific products
- **💬 Natural Language**: Chat with the AI using everyday language
- **📊 Real-time Data**: Up-to-date product information with price updates

## 🏗️ Architecture

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
│  Intent Classifier → Chain Router → Prompt Builder → LLM       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│  API LAYER                                                      │
│  FastAPI → Streaming SSE → Frontend                            │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- Python 3.11+
- OpenAI API key (for embeddings)
- 8GB+ RAM recommended
- 2GB+ disk space for vector store

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the system test**
```bash
python test_system.py
```

## 🚀 Quick Start

### 1. Ingest the Data (First time only)

```bash
python -m src.ingest.embedder
```

This will:
- Load the supermarket dataset (10,000 products)
- Create rich text documents with nutritional information
- Generate embeddings using OpenAI's text-embedding-3-small
- Store in ChromaDB vector store
- Takes ~10 minutes and costs ~$0.01

### 2. Start the API Server

```bash
python -m src.api.main
```

The API will be available at `http://localhost:8000`

### 3. Try the API

```bash
# Health check
curl http://localhost:8000/health

# Chat with the system
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "7-day weight loss meal plan with €60 budget"}'
```

### 4. View API Documentation

Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## 📚 API Endpoints

### `/chat` (POST)
Main chat endpoint with streaming support.

**Request:**
```json
{
  "message": "Give me a high-protein meal plan for €50",
  "conversation_id": "optional-conversation-id"
}
```

**Response:** Streaming JSON with meal plan, costs, and nutritional info.

### `/ingest` (POST)
Trigger data ingestion/re-ingestion.

### `/stats` (GET)
Get database statistics.

### `/update-prices` (POST)
Update product prices (updates both CSV and vector store).

### `/health` (GET)
System health check.

## 💡 Example Queries

### Meal Planning
- "7-day weight loss meal plan, €60 budget"
- "High protein meal plan for muscle gain"
- "Vegan dinner ideas for the week"
- "Keto meal plan under €80"

### Budget Optimization
- "Cheapest high-protein foods"
- "Build a €30 shopping list for a week"
- "Best value breakfast options"
- "Budget-friendly family meals"

### Nutrition Advice
- "Which products have the most protein per euro?"
- "Low calorie snacks under 100 kcal"
- "Best sources of fiber in the store"
- "Compare sugar content in cereals"

### Product Search
- "Italian cheeses with rating above 4"
- "Organic vegetables from Spain"
- "Gluten-free bread options"
- "Products similar to chicken breast"

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
API_PORT=8000
EMBEDDING_BATCH_SIZE=100
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
```

### System Tuning

- **Embedding Batch Size**: Controls API calls (default: 100)
- **Retrieval Top-K**: Initial results (default: 20)
- **Rerank Top-K**: Final results (default: 5)
- **Vector/BM25 Weights**: Hybrid search balance (default: 0.6/0.4)

## 📊 Performance

- **Ingestion**: ~10 minutes for 10,000 products
- **Query Response**: 2-5 seconds (including LLM generation)
- **Memory Usage**: ~2GB for full vector store
- **Cost**: ~$0.01 for initial ingestion, ~$0.001 per 1000 queries

## 🧪 Testing

Run the test suite:

```bash
python test_system.py
```

Tests cover:
- Data ingestion
- Retrieval pipeline
- LLM chains
- API endpoints
- Price updates

## 📁 Project Structure

```
rag-project/
├── data/
│   └── supermarket_dataset.csv          # Source data (10k products)
├── src/
│   ├── ingest/                           # Data ingestion
│   ├── vectorstore/                      # ChromaDB integration
│   ├── retrieval/                        # Search and ranking
│   ├── chains/                          # LLM chains
│   └── api/                             # FastAPI backend
├── tests/                               # Test suite
├── chroma_db/                           # Vector store (auto-created)
├── plan/                                # Planning documents
├── requirements.txt                     # Dependencies
├── .env.example                         # Environment template
└── README.md                            # This file
```

## 🔍 Advanced Features

### Hybrid Retrieval
Combines semantic search (embeddings) with keyword search (BM25) for optimal results.

### HyDE Query Rewriting
Generates hypothetical documents to improve retrieval of vague queries.

### Cross-Encoder Reranking
Uses advanced reranking for better precision on top results.

### Streaming Responses
Real-time streaming of LLM responses for better user experience.

### Price Updates
Live price updates that sync between CSV and vector store with automatic backups.

## 🚨 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY required"**
   - Set your OpenAI API key in `.env` file

2. **"Vector store empty"**
   - Run `python -m src.ingest.embedder` first

3. **"Memory error"**
   - Reduce `EMBEDDING_BATCH_SIZE` in environment
   - Ensure sufficient RAM available

4. **"Slow responses"**
   - Check if ChromaDB is building indices (first few queries are slower)
   - Consider reducing `RETRIEVAL_TOP_K`

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.api.main --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for RAG framework
- **ChromaDB** for vector storage
- **OpenAI** for embeddings and LLM
- **FastAPI** for API framework
- **Sentence Transformers** for reranking

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**Built with ❤️ using RAG technology**
