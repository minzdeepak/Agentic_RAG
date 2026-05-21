"""
Quick Start Guide for RAG System
"""

# 🚀 QUICK START GUIDE

# ============================================
# Step 1: Installation
# ============================================

# Option A: Local Installation
"""
1. Create virtual environment:
   python -m venv venv
   
2. Activate environment:
   # Windows:
   venv\\Scripts\\activate
   # macOS/Linux:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Start Ollama:
   ollama serve
   # In another terminal:
   ollama pull llama3.1
"""

# Option B: Docker Installation
"""
docker-compose up --build
"""


# ============================================
# Step 2: Configure Environment
# ============================================

"""
Copy and edit .env file:

cp .env.example .env

Update with your settings:
- OLLAMA_BASE_URL=http://192.168.6.79:11434
- EMBEDDING_DEVICE=cuda (or cpu)
- LOG_LEVEL=INFO
"""


# ============================================
# Step 3: Start the System
# ============================================

"""
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start RAG System
python main.py

# API available at: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
"""


# ============================================
# Step 4: Ingest Your First Document
# ============================================

"""
# Using curl:
curl -X POST "http://localhost:8000/api/v1/ingest" \\
  -F "file=@your_document.pdf"

# Response will include document_id and chunks_created
"""


# ============================================
# Step 5: Query the System
# ============================================

"""
# Using curl:
curl -X POST "http://localhost:8000/api/v1/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is the main topic?",
    "top_k": 3,
    "include_sources": true,
    "use_hybrid_search": true
  }'

# Get answer from LLM based on document context
"""


# ============================================
# Quick Test Script
# ============================================

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def quick_test():
    """Run a quick test of the system"""
    
    print("🚀 RAG System Quick Test\n")
    
    try:
        # Test 1: Load embedding model
        print("1️⃣  Testing Embeddings...")
        from embeddings.embedding_model import EmbeddingModel
        embedding = EmbeddingModel()
        test_emb = embedding.embed("Hello world")
        print(f"   ✅ Embedding dimension: {len(test_emb)}")
        
        # Test 2: Check vector database
        print("\n2️⃣  Testing Vector Database...")
        from vectordb.chromadb_store import ChromaVectorStore
        vs = ChromaVectorStore()
        info = vs.get_collection_info()
        print(f"   ✅ Vector DB ready. Documents: {info['document_count']}")
        
        # Test 3: Check LLM
        print("\n3️⃣  Testing LLM Connection...")
        from llm.ollama_llm import OllamaLLM
        llm = OllamaLLM()
        models = llm.list_available_models()
        print(f"   ✅ Ollama connected. Models: {models}")
        
        # Test 4: Health check
        print("\n4️⃣  System Health Check...")
        print("   ✅ All components initialized successfully!")
        
        print("\n✨ System is ready to use!")
        print("\nNext steps:")
        print("1. Ingest documents: POST /api/v1/ingest")
        print("2. Query documents: POST /api/v1/query")
        print("3. Check docs: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("- Make sure Ollama is running: ollama serve")
        print("- Check environment variables in .env")
        print("- Review logs for detailed errors")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
