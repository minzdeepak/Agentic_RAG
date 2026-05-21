# 🚀 Production-Grade RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, and modern AI technologies. This system integrates document processing, semantic chunking, hybrid search, and LLM-based answer generation.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Switching LLM Providers](#switching-llm-providers)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Architecture](#retrieval-architecture)
- [Docker Deployment](#docker-deployment)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

This RAG system implements a complete pipeline from document ingestion to answer generation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG SYSTEM PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DOCUMENT INGESTION                                            │
│  ├── File Upload (PDF, DOCX, DOC, TXT)                        │
│  ├── Unstructured.io Processing                               │
│  └── Metadata Extraction                                       │
│           ↓                                                    │
│  DATA CLEANING                                                 │
│  ├── Text Normalization                                        │
│  ├── Boilerplate Removal                                       │
│  └── Special Character Handling                                │
│           ↓                                                    │
│  HYBRID SEMANTIC CHUNKING                                      │
│  ├── Structure Detection (Headings, Lists, Tables)             │
│  ├── Section Segmentation                                      │
│  ├── Semantic Boundary Detection                               │
│  └── Token-Aware Chunk Merge (300-800 tokens)                  │
│           ↓                                                    │
│  EMBEDDING GENERATION                                          │
│  └── BAAI/bge-m3 Model                             │
│           ↓                                                    │
│  VECTOR STORAGE                                                │
│  └── ChromaDB (Persistent Storage)                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  RETRIEVAL & GENERATION                                         │
│                                                                 │
│  QUERY PROCESSING                                              │
│  └── Embedding Generation                                      │
│           ↓                                                    │
│  HYBRID SEARCH                                                 │
│  ├── Dense Vector Search (Cosine Similarity)                  │
│  └── BM25 Keyword Search                                       │
│           ↓                                                    │
│  RESULT MERGING                                                │
│  └── Weighted Score Combination (α = 0.5)                      │
│           ↓                                                    │
│  RERANKING                                                      │
│  └── Cross-Encoder Model (Optional)                            │
│           ↓                                                    │
│  TOP K SELECTION                                               │
│  └── Final Context Selection (Top 3)                           │
│           ↓                                                    │
│  LLM ANSWER GENERATION                                         │
│  └── Pluggable LLM Provider (Ollama · Gemini · OpenAI · Claude)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI 0.104.1 | RESTful API endpoints |
| **Document Processing** | Unstructured.io 0.14.10 | Multi-format document parsing |
| **Text Chunking** | Custom Semantic Chunker | Intelligent content segmentation |
| **Embeddings** | Sentence-Transformers 2.6.1 + BAAI/bge-m3 | Dense vector generation |
| **Vector Database** | ChromaDB 0.5.3 (PersistentClient) | Persistent vector storage |
| **Retrieval** | BM25 + Dense Search | Hybrid search mechanism |
| **LLM** | Ollama (default) · Gemini · OpenAI · Claude | Pluggable LLM inference via `LLM_PROVIDER` |
| **Reranking** | BAAI/bge-reranker-large | Result optimization |
| **Containerization** | Docker + Docker Compose | Production deployment |

### Key Dependencies

```
Python 3.11+
FastAPI 0.104.1
Sentence-Transformers 2.6.1
ChromaDB 0.5.3 (requires PersistentClient API)
Torch 2.2.1
Unstructured 0.14.10 (with pdf, docx extras)
Python-DOCX 1.1.2
Pdfminer-six 20221105
Rank-BM25 0.2.2
```

---

## 📁 Project Structure

```
rag-system/
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI application
│
├── ingestion/
│   ├── __init__.py
│   ├── document_processor.py       # Unstructured.io integration
│   ├── data_cleaner.py             # Data cleaning & preprocessing
│   └── ingestion_pipeline.py       # Complete ingestion workflow
│
├── chunking/
│   ├── __init__.py
│   └── semantic_chunker.py         # Hybrid semantic chunking algorithm
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py          # BAAI embedding model wrapper
│
├── vectordb/
│   ├── __init__.py
│   └── chromadb_store.py           # ChromaDB integration
│
├── retrieval/
│   ├── __init__.py
│   └── hybrid_retriever.py         # Hybrid search + reranking
│
├── llm/
│   ├── __init__.py
│   ├── base_llm.py                # Abstract base class for all LLM providers
│   ├── ollama_llm.py              # Ollama LLM integration (default)
│   ├── gemini_llm.py              # Google Gemini integration
│   ├── openai_llm.py              # OpenAI integration
│   ├── claude_llm.py              # Anthropic Claude integration
│   └── llm_factory.py             # Provider factory — reads LLM_PROVIDER env var
│
├── models/
│   ├── __init__.py
│   └── schemas.py                  # Pydantic data models
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # Logging configuration
│   └── helpers.py                  # Utility functions
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration management
│
├── main.py                         # Application entry point
├── requirements.txt                # Core Python dependencies
├── requirements-llm-providers.txt  # Optional LLM provider SDKs
├── Dockerfile                      # Container image
├── docker-compose.yml              # Multi-service orchestration
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running (or accessible via network) — required only when `LLM_PROVIDER=ollama`
- **CUDA 11.8+** (optional, for GPU acceleration)

### Option 1: Local Installation

#### 1. Clone and Setup

```bash
cd rag-system
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Install LLM Provider SDK (if not using Ollama)

```bash
# Install all optional provider SDKs at once:
pip install -r requirements-llm-providers.txt

# Or install only the one you need:
pip install google-generativeai>=0.8.0   # for LLM_PROVIDER=gemini
pip install openai>=1.40.0               # for LLM_PROVIDER=openai
pip install anthropic>=0.34.0            # for LLM_PROVIDER=claude
```

#### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings — set LLM_PROVIDER and the matching API key
```

#### 5. Start Ollama (Ollama provider only)

```bash
# Skip this step if using Gemini, OpenAI, or Claude
ollama serve

# In another terminal, pull the model:
ollama pull llama3.1
```

#### 6. Run Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Installation

#### 1. Clone and Navigate

```bash
cd rag-system
```

#### 2. Build and Start Services

```bash
docker-compose up --build
```

#### 3. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Ollama
curl http://ollama:11434/api/tags
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda  # or 'cpu'
EMBEDDING_BATCH_SIZE=32

# Vector Database
CHROMA_COLLECTION_NAME=rag_documents

# Retrieval Parameters
TOP_K_DENSE=5              # Dense search results
TOP_K_BM25=5               # BM25 search results
TOP_K_FINAL=3              # Final top-k results
RERANKER_MODEL=BAAI/bge-reranker-large

# LLM Provider Selection — ollama | gemini | openai | claude
LLM_PROVIDER=ollama

# LLM Configuration (Ollama — default)
OLLAMA_BASE_URL=http://192.168.6.79:11434
OLLAMA_MODEL=llama3.1
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_TIMEOUT=60

# LLM Configuration (Gemini) — set when LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.7

# LLM Configuration (OpenAI) — set when LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7

# LLM Configuration (Claude) — set when LLM_PROVIDER=claude
CLAUDE_API_KEY=
CLAUDE_MODEL=claude-sonnet-4-6
CLAUDE_TEMPERATURE=0.7
CLAUDE_MAX_TOKENS=4096

# Logging
LOG_LEVEL=INFO
```

### Switching LLM Providers

Change only one environment variable to switch providers:

| `LLM_PROVIDER` | Required env vars | SDK to install |
|---|---|---|
| `ollama` (default) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | none (uses HTTP) |
| `gemini` | `GEMINI_API_KEY` | `pip install google-generativeai>=0.8.0` |
| `openai` | `OPENAI_API_KEY` | `pip install openai>=1.40.0` |
| `claude` | `CLAUDE_API_KEY` | `pip install anthropic>=0.34.0` |

### Settings Customization

Edit `config/settings.py` for advanced configuration:

```python
# Chunking Parameters
MIN_CHUNK_SIZE = 300           # Minimum tokens per chunk
MAX_CHUNK_SIZE = 800           # Maximum tokens per chunk
OVERLAP = 50                   # Token overlap between chunks
SIMILARITY_THRESHOLD = 0.7     # Semantic boundary detection

# File Upload
ALLOWED_FILE_TYPES = {"pdf", "docx", "doc", "txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
```

---

## 📚 Usage

### 1. Ingest Documents

#### Using Python

```python
from ingestion.ingestion_pipeline import IngestionPipeline
from embeddings.embedding_model import EmbeddingModel
from vectordb.chromadb_store import ChromaVectorStore
from models.schemas import SourceType

# Initialize components
embedding_model = EmbeddingModel()
vector_store = ChromaVectorStore()
pipeline = IngestionPipeline(embedding_model, vector_store)

# Ingest document
result = pipeline.ingest_document(
    file_path="document.pdf",
    source_type=SourceType.PDF,
    document_metadata={"category": "finance"}
)

print(f"Document ID: {result['document_id']}")
print(f"Chunks created: {result['chunks_created']}")
```

#### Using REST API

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@document.pdf" \
  -H "Content-Type: multipart/form-data"
```

### 2. Query the System

#### Using Python

```python
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_factory import get_llm

# Initialize retriever and LLM (provider chosen via LLM_PROVIDER env var)
retriever = HybridRetriever(embedding_model, vector_store)
llm = get_llm()

# Retrieve documents and generate answer
question = "What are the key points in the document?"
retrieved_docs = retriever.retrieve(question, top_k=3)

answer = llm.generate_answer(
    query=question,
    context=[doc['content'] for doc in retrieved_docs]
)

print(answer['answer'])
```

#### Using REST API

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key points?",
    "top_k": 3,
    "include_sources": true,
    "use_hybrid_search": true
  }'
```

---

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "vectordb": "ready",
  "embeddings": "ready",
  "llm": "ready",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. Ingest Document

```http
POST /api/v1/ingest
Content-Type: multipart/form-data

file: <binary>
document_metadata: {"category": "finance"}  # Optional JSON
```

**Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "document.pdf",
  "chunks_created": 15,
  "status": "success",
  "message": "Successfully ingested 15 chunks",
  "created_at": "2024-01-15T10:30:00"
}
```

#### 3. Query RAG System

```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "What are the main topics?",
  "top_k": 3,
  "include_sources": true,
  "use_hybrid_search": true
}
```

**Response:**
```json
{
  "question": "What are the main topics?",
  "answer": "The main topics covered are...",
  "sources": [
    {
      "content": "Topic 1: ...",
      "score": 0.85,
      "metadata": {}
    }
  ],
  "confidence": 0.82,
  "processing_time": 2.34,
  "model": "BAAI/bge-m3",
  "llm_model": "llama3.1"
}
```

#### 4. Get Vector Database Info

```http
GET /api/v1/vectordb/info
```

**Response:**
```json
{
  "name": "rag_documents",
  "document_count": 45,
  "metadata": {}
}
```

#### 5. Get Models Information

```http
GET /api/v1/models
```

**Response (Ollama provider example):**
```json
{
  "embedding": {
    "model_name": "BAAI/bge-m3",
    "embedding_dimension": 1024,
    "device": "cuda",
    "max_seq_length": 512
  },
  "llm": {
    "provider": "ollama",
    "model": "llama3.1",
    "base_url": "http://192.168.6.79:11434",
    "temperature": 0.7,
    "top_p": 0.9,
    "timeout": 60
  },
  "available_llm_models": ["llama3.1", "mistral"]
}
```

#### 6. Clear Vector Database

```http
DELETE /api/v1/vectordb/clear
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector database cleared"
}
```

---

## 🔀 Chunking Strategy

### Hybrid Semantic Chunking Algorithm

The system implements a four-step chunking process:

#### Step 1: Structural Parsing

Identifies and extracts:
- **Headings** (Markdown # or uppercase text)
- **Paragraphs** (Regular text blocks)
- **Bullet Lists** (- • * prefixed lines)
- **Numbered Lists** (1. 2. 3. prefixed lines)
- **Tables** (| or \t delimited content)

```python
Document Text
    ↓
Element Detection
    ↓
[Heading] Chapter 1
[Paragraph] Introduction text...
[Bullet] - First point
[Bullet] - Second point
[Table] | Header | Value |
```

#### Step 2: Section Segmentation

Groups elements into logical sections, typically delimited by headings:

```
Section 1 (Heading 1)
├── Paragraph 1
├── Paragraph 2
└── Bullet List 1

Section 2 (Heading 2)
├── Paragraph 3
└── Table 1
```

#### Step 3: Semantic Boundary Detection

Uses embedding similarity to detect semantic boundaries between adjacent paragraphs:

```python
Paragraph A: "Introduction to machine learning..."
    ↓
Embedding A: [0.12, 0.45, -0.33, ...]

Paragraph B: "Deep learning architectures..."
    ↓
Embedding B: [0.18, 0.42, -0.31, ...]

Cosine Similarity: 0.92 (> 0.7 threshold)
    ↓
Result: Same semantic section, keep together
```

#### Step 4: Token-Aware Chunk Merge

Merges segments while maintaining constraints:

```
Constraints:
├── MIN_CHUNK_SIZE: 300 tokens
├── MAX_CHUNK_SIZE: 800 tokens
└── OVERLAP: 50 tokens between chunks

Algorithm:
1. Start with first segment
2. Add segments while total < MAX_CHUNK_SIZE
3. If adding would exceed MAX, save chunk if >= MIN_CHUNK_SIZE
4. Create overlap context for next chunk
5. Repeat for all sections
```

### Example Output

```json
{
  "chunk_index": 0,
  "content": "Chapter 1: Introduction\n\nThis chapter covers...",
  "token_count": 450,
  "char_count": 2100,
  "sentences": 8
}
```

---

## 🔍 Retrieval Architecture

### Query Processing Pipeline

```
User Query
    ↓
Step 1: Generate Query Embedding
    ├── Input: "What is machine learning?"
    ├── Model: BAAI/bge-m3
    └── Output: [0.23, -0.45, 0.12, ...] (1024 dimensions)
    ↓
Step 2: Dense Vector Search
    ├── Method: Cosine Similarity
    ├── Top-K: 5 results
    └── Scores: [0.91, 0.87, 0.82, 0.75, 0.71]
    ↓
Step 3: BM25 Keyword Search
    ├── Method: Term Frequency Analysis
    ├── Top-K: 5 results
    └── Scores: [145.2, 132.8, 98.5, 87.3, 76.1]
    ↓
Step 4: Merge Results
    ├── Normalize Dense Scores: [1.0, 0.956, 0.901, 0.824, 0.781]
    ├── Normalize BM25 Scores: [1.0, 0.915, 0.679, 0.602, 0.525]
    ├── Alpha (weight): 0.5
    ├── Combined = 0.5 * dense + 0.5 * bm25
    └── Merged Results: [{score: 0.978}, {score: 0.936}, ...]
    ↓
Step 5: Rerank (Optional)
    ├── Model: BAAI/bge-reranker-large
    ├── Cross-Encoder scoring
    └── Reranked Results
    ↓
Step 6: Top-K Selection
    └── Top 3 Results
    ↓
Step 7: Context Building
    ├── Result 1: "Machine learning is..."
    ├── Result 2: "Types of ML include..."
    └── Result 3: "Applications of ML..."
    ↓
Step 8: LLM Answer Generation
    ├── Prompt: [System] + [Context] + [Question]
    ├── Provider: Ollama | Gemini | OpenAI | Claude (via LLM_PROVIDER)
    └── Answer: "Based on the provided context..."
```

### Hybrid Search Mechanism

#### Dense Vector Search (60% weight in example)

```python
# Cosine similarity: higher = more similar
similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))

# Score range: -1 to 1 (typically 0 to 1 for positive matches)
score = (similarity + 1) / 2  # Normalize to 0-1
```

#### BM25 Keyword Search (40% weight in example)

```python
# Term Frequency - Inverse Document Frequency
# Considers:
# - How often term appears in document
# - How rare the term is across all documents
# - Document length normalization

score = IDF(term1) * (f(term1, doc) * (k1 + 1)) / 
        (f(term1, doc) + k1 * (1 - b + b * len(doc) / avg_doc_len))
```

#### Score Combination

```python
alpha = 0.5  # Weight for dense search

# Normalize scores to 0-1 range
dense_score_normalized = dense_score / max(dense_scores)
bm25_score_normalized = bm25_score / max(bm25_scores)

# Combine with alpha weighting
combined_score = alpha * dense_score_normalized + (1 - alpha) * bm25_score_normalized
```

---

## 🐳 Docker Deployment

### Quick Start

```bash
cd rag-system
docker-compose up --build
```

### Important: ChromaDB Migration to v0.5.3

The system has been updated to use **ChromaDB 0.5.3** with the new **PersistentClient API**.

If you have existing ChromaDB data from older versions:

```bash
# Option 1: Use migration tool (if you have important data)
pip install chroma-migrate
chroma-migrate

# Option 2: Fresh start (if starting new)
# Simply clear your chroma_db directory and start fresh:
rm -rf chroma_db/*
docker-compose up --build
```

### Services

1. **rag-api** (Port 8000)
   - FastAPI application
   - Mounts: uploads, chroma_db, logs
   - Now compatible with ChromaDB 0.5.3 PersistentClient

2. **ollama** (Port 11434)
   - Ollama LLM server
   - Mount: ollama_data volume

### Environment Customization

Edit `docker-compose.yml` before building:

```yaml
services:
  rag-api:
    environment:
      - EMBEDDING_DEVICE=cuda  # For GPU
      - OLLAMA_BASE_URL=http://ollama:11434
      - LOG_LEVEL=DEBUG
```

### Build Images Separately

```bash
# Build RAG API image
docker build -t rag-system:latest .

# Run RAG API
docker run -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rag-system:latest

# Run Ollama
docker run -d -p 11434:11434 ollama/ollama:latest
ollama pull llama3.1
```

### Verify Deployment

```bash
# Check API
curl http://localhost:8000/health

# Check Ollama
curl http://localhost:11434/api/tags

# Check logs
docker logs rag-system-api
docker logs ollama-server
```

---

## ⚡ Performance Considerations

### Optimization Tips

#### 1. Chunking

```python
# Config optimizations
MIN_CHUNK_SIZE = 300      # Larger = fewer chunks but more context
MAX_CHUNK_SIZE = 800      # Affects token per request
OVERLAP = 50              # More overlap = better context but slower retrieval
```

#### 2. Embedding

```python
# Use CUDA for GPU acceleration
EMBEDDING_DEVICE = "cuda"  # vs "cpu"
EMBEDDING_BATCH_SIZE = 32  # Increase for better throughput
```

#### 3. Retrieval

```python
# Adjust K values based on requirements
TOP_K_DENSE = 5    # Increase for better recall
TOP_K_BM25 = 5     # Increase for keyword coverage
TOP_K_FINAL = 3    # Balance between quality and speed
```

#### 4. LLM

```python
# Select provider via LLM_PROVIDER=ollama|gemini|openai|claude
# Ollama (local — no API cost, depends on hardware)
OLLAMA_TEMPERATURE = 0.7     # Lower = more deterministic
OLLAMA_TOP_P = 0.9           # Lower = more focused
OLLAMA_TIMEOUT = 60          # Seconds

# Cloud providers (Gemini / OpenAI / Claude)
# Lower temperature → more consistent answers
GEMINI_TEMPERATURE = 0.7
OPENAI_TEMPERATURE = 0.7
CLAUDE_TEMPERATURE = 0.7
CLAUDE_MAX_TOKENS = 4096     # Claude requires an explicit max_tokens limit
```

### Benchmarks

| Operation | Time (avg) | Notes |
|-----------|-----------|-------|
| Document Ingestion (10 MB PDF) | 2-5 min | Depends on content density |
| Query Processing | 2-4 sec | Includes embedding + retrieval + LLM |
| Embedding Generation | 50-100 ms | For 1 chunk, CUDA device |
| Dense Search | 10-50 ms | Depends on collection size |
| BM25 Search | 5-20 ms | Very fast keyword matching |
| LLM Generation | 1-3 sec | Model + hardware dependent |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

```
Error: Could not connect to Ollama at http://192.168.6.79:11434
```

**Solution:**
```bash
# Check if Ollama is running
curl http://192.168.6.79:11434/api/tags

# If using Docker, ensure proper network:
docker network inspect rag-network

# Update OLLAMA_BASE_URL in .env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Switch to CPU
EMBEDDING_DEVICE=cpu

# Or reduce batch size
EMBEDDING_BATCH_SIZE=8

# Or clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. Model Download Timeout

```
Error: Failed to download BAAI/bge-m3
```

**Solution:**
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Or use offline mode
HF_DATASETS_OFFLINE=1 python main.py
```

#### 4. Empty Search Results

```
HTTPException: No relevant documents found for this query
```

**Solution:**
```python
# Check vector database
from vectordb.chromadb_store import ChromaVectorStore
vs = ChromaVectorStore()
info = vs.get_collection_info()
print(f"Documents: {info['document_count']}")

# Ingest more documents
# Adjust search parameters
TOP_K_FINAL = 5

# Check query quality
# Rephrase question for better matching
```

#### 5. Slow Retrieval

**Solution:**
```python
# Profile the pipeline
import time

start = time.time()
results = retriever.retrieve(query, top_k=3)
print(f"Dense search: {time.time() - start:.2f}s")

# Reduce TOP_K values
TOP_K_DENSE = 3
TOP_K_BM25 = 3

# Use GPU for embeddings
EMBEDDING_DEVICE = "cuda"

# Disable reranking if slow
RERANKER_MODEL = None
```

#### 6. ChromaDB Deprecation Error

```
ValueError: You are using a deprecated configuration of Chroma.
```

**Solution:**
The system now uses the new ChromaDB PersistentClient API. If you encounter this error:

```bash
# Clear old ChromaDB data
rm -rf chroma_db/*

# Rebuild and restart
docker-compose build --no-cache
docker-compose up
```

Or if you have important data, use the migration tool:

```bash
pip install chroma-migrate
chroma-migrate
```

#### 7. LLM Provider SDK Not Installed

```
ImportError: google-generativeai is not installed. Run: pip install google-generativeai>=0.8.0
ImportError: openai is not installed. Run: pip install openai>=1.40.0
ImportError: anthropic is not installed. Run: pip install anthropic>=0.34.0
```

**Solution:**
```bash
# Install the required SDK for your chosen provider
pip install google-generativeai>=0.8.0   # gemini
pip install openai>=1.40.0               # openai
pip install anthropic>=0.34.0            # claude

# Or install all at once
pip install -r requirements-llm-providers.txt
```

#### 8. Missing or Invalid API Key

```
ValueError: GEMINI_API_KEY is required when LLM_PROVIDER=gemini
ValueError: OPENAI_API_KEY is required when LLM_PROVIDER=openai
ValueError: CLAUDE_API_KEY is required when LLM_PROVIDER=claude
```

**Solution:**
```env
# In .env — set the key for the provider you selected
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
```

#### 9. Unknown LLM Provider

```
ValueError: Unknown LLM_PROVIDER 'xyz'. Valid values: ollama | gemini | openai | claude
```

**Solution:** Set `LLM_PROVIDER` to one of the four supported values in `.env`.

#### 10. Dependency Conflicts (requests/urllib3/charset-normalizer)

```
RequestsDependencyWarning: urllib3 or chardet/charset_normalizer doesn't match
```

**Solution:**
The system now uses compatible versions:
- `requests==2.31.0`
- `urllib3==2.0.7`
- `chardet==5.2.0`

These are automatically installed. If issues persist:

```bash
docker-compose build --no-cache
docker-compose up
```

### Logging

Enable detailed logging for debugging:

```env
LOG_LEVEL=DEBUG
```

Logs are stored in:
- Console: stdout/stderr
- File: `logs/` directory

### Health Checks

```bash
# Full system health
curl http://localhost:8000/health

# Component info
curl http://localhost:8000/api/v1/models

# Vector DB status
curl http://localhost:8000/api/v1/vectordb/info
```

---

## 📊 Metadata Storage Design

### Chunk Metadata Structure

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "financial_report.pdf",
  "page": 5,
  "section": "Financial Analysis",
  "chunk_id": "550e8400-e29b-41d4-a716-446655440000_5",
  "chunk_index": 5,
  "source_type": "pdf",
  "token_count": 450,
  "char_count": 2100,
  "created_at": "2024-01-15T10:30:00.000Z",
  "updated_at": "2024-01-15T10:30:05.000Z",
  "embedding_model": "BAAI/bge-m3",
  "custom_field": "custom_value"
}
```

### Metadata Filtering

```python
# Filter by document_id
results = vector_store.search(
    query_embedding,
    where={"document_id": "550e8400-e29b-41d4-a716-446655440000"}
)

# Filter by source_type
results = vector_store.search(
    query_embedding,
    where={"source_type": "pdf"}
)

# Complex filtering
results = vector_store.search(
    query_embedding,
    where={
        "$and": [
            {"source_type": "pdf"},
            {"token_count": {"$gte": 300}}
        ]
    }
)
```

---

## 🎯 Next Steps & Extensions

### Potential Enhancements

1. **Advanced Chunking**
   - Dynamic chunk sizing based on semantic density
   - Multi-level hierarchical chunking
   - Custom chunking strategies per document type

2. **Retrieval Improvements**
   - Query expansion and reformulation
   - Document routing (specialized retrievers)
   - Iterative retrieval with refinement

3. **RAG Optimization**
   - Answer quality scoring
   - Confidence calibration
   - Hallucination detection

4. **Scalability**
   - Database sharding
   - Distributed embeddings
   - Caching layer (Redis)

5. **Monitoring**
   - Query analytics
   - Performance metrics
   - Cost tracking

---

## 📝 License

This project is provided as-is for educational and commercial use.

---

## 👥 Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review configuration in `config/settings.py`
3. Enable DEBUG logging for detailed diagnostics
4. Check component logs in the `logs/` directory

---

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Unstructured.io](https://unstructured.io/)
- [Ollama](https://ollama.ai/)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude API](https://docs.anthropic.com/en/api/getting-started)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [RAG Papers](https://arxiv.org/search/?query=retrieval+augmented+generation)

---

## 🔄 Recent Updates (May 2026)

### Version 1.2.0 - Multi-LLM Provider Support

#### New: Pluggable LLM Providers
- Added support for **Google Gemini**, **OpenAI**, and **Anthropic Claude** alongside existing Ollama
- Single `LLM_PROVIDER` environment variable switches the active provider — no code changes required
- Default remains `ollama` for zero breaking-change upgrades

#### New Files
- `llm/base_llm.py` — Abstract base class; shared `_build_prompt` and `generate_answer` logic
- `llm/gemini_llm.py` — Google Gemini integration (`gemini-2.0-flash` default)
- `llm/openai_llm.py` — OpenAI integration (`gpt-4o` default)
- `llm/claude_llm.py` — Anthropic Claude integration (`claude-sonnet-4-6` default)
- `llm/llm_factory.py` — `get_llm()` factory; provider SDKs loaded lazily (only when selected)
- `requirements-llm-providers.txt` — Optional SDK pinning for all three cloud providers

#### Refactored
- `llm/ollama_llm.py` now inherits `BaseLLM`; shared prompt-building logic moved to base class
- `api/main.py` uses `get_llm()` factory instead of hard-coded `OllamaLLM`

#### Configuration additions
- New env vars: `LLM_PROVIDER`, `GEMINI_*`, `OPENAI_*`, `CLAUDE_*` (see `.env.example`)

---

### Version 1.1.0 - Stability & Compatibility Improvements

#### ChromaDB Migration
- **Migrated to ChromaDB 0.5.3** with new `PersistentClient` API
- Removed deprecated `Settings` configuration
- Improved persistence and reliability

#### Dependency Updates
- Updated Unstructured to 0.14.10 for better document processing
- Python-DOCX 1.1.2 for DOCX support
- Added pdfminer-six 20221105 and pdfplumber 0.10.4
- Fixed urllib3/charset-normalizer compatibility issues
- Updated Sentence-Transformers to 2.6.1

#### Docker Improvements
- Added missing OpenCV dependencies (libgl1, libsm6, etc.)
- Increased pip timeout and retry settings for reliability
- Optimized layer caching for faster builds
- Configurable reload mode for API

#### Bug Fixes
- Fixed missing docx.text.hyperlink import
- Resolved OpenCV library loading errors
- Fixed requests dependency warnings
- Improved system library support in Docker

---

**Last Updated:** May 2026  
**Version:** 1.2.0
