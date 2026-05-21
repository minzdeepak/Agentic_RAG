# 🏗️ RAG System Architecture Documentation

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL-AUGMENTED GENERATION SYSTEM              │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐       │
│  │  REST API       │  │  Python SDK     │  │  Web Dashboard   │       │
│  │  (FastAPI)      │  │  (Direct Call)  │  │  (Optional)      │       │
│  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘       │
│           │                    │                    │                  │
│           └────────────────────┼────────────────────┘                  │
│                                │                                       │
└────────────────────────────────┼───────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   API Layer (FastAPI)   │
                    │  ┌──────────────────┐  │
                    │  │ /api/v1/ingest   │  │
                    │  │ /api/v1/query    │  │
                    │  │ /health          │  │
                    │  │ /models          │  │
                    │  └──────────────────┘  │
                    └────┬────────────┬─────┘
                         │            │
             ┌───────────▼──┐   ┌────▼──────────┐
             │  Ingestion   │   │  Retrieval    │
             │  Pipeline    │   │  Pipeline     │
             └───────┬──────┘   └───┬──────┬────┘
                     │              │      │
        ┌────────────┼──────────────┤      │
        │            │              │      │
        ▼            ▼              ▼      ▼
    ┌─────────────────────────┐  ┌──────────────┐
    │  INGESTION PIPELINE     │  │  RETRIEVAL   │
    ├─────────────────────────┤  └──────────────┘
    │ 1. Document Upload      │
    │    • PDF, DOCX, DOC     │
    │    • Size Validation    │
    │                         │
    │ 2. Document Processing  │
    │    • Unstructured.io    │
    │    • Structure Extract  │
    │    • Element Detection  │
    │                         │
    │ 3. Data Cleaning        │
    │    • URL Removal        │
    │    • Boilerplate Strip  │
    │    • Normalization      │
    │                         │
    │ 4. Semantic Chunking    │
    │    • Struct Parsing     │
    │    • Section Segment    │
    │    • Boundary Detect    │
    │    • Token Merge        │
    │                         │
    │ 5. Embedding Gen        │
    │    • BAAI/bge-v1.5      │
    │    • 1024-dim vectors   │
    │    • Batch Processing   │
    │                         │
    │ 6. Storage              │
    │    • ChromaDB           │
    │    • Metadata Store     │
    └────────────────┬────────┘
                     │
        ┌────────────▼───────────────────────────┐
        │    VECTOR DATABASE (ChromaDB)          │
        ├─────────────────────────────────────────┤
        │  Collection: rag_documents              │
        │  ┌──────────────────────────────────┐  │
        │  │ Documents                        │  │
        │  │ ├─ chunk_id (ID)                │  │
        │  │ ├─ content (text)               │  │
        │  │ ├─ embedding (1024-dim)         │  │
        │  │ └─ metadata (document info)     │  │
        │  └──────────────────────────────────┘  │
        │  Storage: Persistent (DuckDB)          │
        │  Indexing: HNSW (Approximate NN)       │
        └────────────────┬───────────────────────┘
                         │
        ┌────────────────▼───────────────────────┐
        │    RETRIEVAL PIPELINE                  │
        ├─────────────────────────────────────────┤
        │ 1. Query Processing                     │
        │    └─ Generate Query Embedding          │
        │                                         │
        │ 2. Dense Vector Search                  │
        │    ├─ Cosine Similarity                │
        │    ├─ Top-K: 5 results                 │
        │    └─ Score: 0-1 range                 │
        │                                         │
        │ 3. BM25 Keyword Search                  │
        │    ├─ Term Frequency Analysis          │
        │    ├─ Top-K: 5 results                 │
        │    └─ Fast keyword matching            │
        │                                         │
        │ 4. Result Merging                       │
        │    ├─ Normalize scores (0-1)           │
        │    ├─ Alpha weighting (0.5)            │
        │    └─ Combined score: 0.5*dense + 0.5*bm25
        │                                         │
        │ 5. Reranking (Optional)                 │
        │    ├─ CrossEncoder model               │
        │    └─ Context-aware ranking            │
        │                                         │
        │ 6. Top-K Selection                      │
        │    └─ Final: Top 3 results             │
        │                                         │
        │ 7. Context Building                     │
        │    └─ Combine doc content              │
        └────────────────┬───────────────────────┘
                         │
        ┌────────────────▼───────────────────────┐
        │    LLM INTEGRATION (Ollama)            │
        ├─────────────────────────────────────────┤
        │ Model: llama3.1                         │
        │ ┌──────────────────────────────────┐   │
        │ │ Prompt Engineering               │   │
        │ │ ├─ System Prompt                 │   │
        │ │ ├─ Context Injection             │   │
        │ │ ├─ Question Integration          │   │
        │ │ └─ Few-shot Examples             │   │
        │ └──────────────────────────────────┘   │
        │ Temperature: 0.7                        │
        │ Top-P: 0.9                              │
        │ Response: Streamed/Buffered             │
        └────────────────┬───────────────────────┘
                         │
        ┌────────────────▼───────────────────────┐
        │    RESPONSE GENERATION                  │
        ├─────────────────────────────────────────┤
        │ • Answer Text                           │
        │ • Source Documents                      │
        │ • Confidence Score                      │
        │ • Processing Time                       │
        │ • Model Information                     │
        └─────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
                    INPUT
                     │
        ┌────────────▼────────────┐
        │  Document Upload        │
        │  (REST API)             │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────┐
        │ Unstructured.io Processor   │
        │ ├─ Format Detection         │
        │ ├─ OCR (if needed)          │
        │ ├─ Table Extraction         │
        │ └─ Element Parsing          │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ Text Cleaning               │
        │ ├─ URL/Email Removal        │
        │ ├─ Whitespace Norm          │
        │ ├─ Boilerplate Strip        │
        │ └─ Special Char Handling    │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ Semantic Chunking           │
        │ ├─ Structure Detection      │
        │ ├─ Section Segmentation     │
        │ ├─ Boundary Detection       │
        │ └─ Token Merging            │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ Embedding Generation        │
        │ (BAAI/bge-m3)   │
        │ ├─ Batch Processing         │
        │ ├─ GPU Acceleration         │
        │ └─ Dimension: 1024          │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │ Vector Storage              │
        │ ChromaDB (DuckDB)           │
        │ ├─ Document Indexing        │
        │ ├─ Metadata Storage         │
        │ └─ Persistence              │
        └────────────┬────────────────┘
                     │
                 STORAGE
```

---

## Component Interactions

```
┌──────────────────┐
│   FastAPI API    │
└────────┬─────────┘
         │
         ├────────────────────────────┐
         │                            │
         ▼                            ▼
   ┌──────────────┐         ┌──────────────────┐
   │  Ingestion   │         │   Retrieval      │
   │  Pipeline    │         │   Pipeline       │
   └──────┬───────┘         └────────┬─────────┘
          │                         │
          │                    ┌────▼────┐
          │                    │ Embedder │
          │                    └────┬────┘
          │                         │
          ├─────────┬───────────────┘
          │         │
          ▼         ▼
    ┌──────────────────────┐
    │  ChromaDB Vector DB  │
    └──────────┬───────────┘
               │
         ┌─────▼─────┐
         │  Retriever│
         ├─ Dense    │
         ├─ BM25     │
         └─ Reranker │
         └─────┬─────┘
               │
         ┌─────▼──────┐
         │ Ollama LLM │
         │ llama3.1   │
         └────────────┘
```

---

## Technology Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Presentation Layer                                          │
├─────────────────────────────────────────────────────────────┤
│  REST API (FastAPI) | Swagger UI | Direct Python API       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Application Layer                                           │
├─────────────────────────────────────────────────────────────┤
│  • Ingestion Service                                        │
│  • Retrieval Service                                        │
│  • LLM Service                                              │
│  • Vector DB Service                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Processing Layer                                            │
├─────────────────────────────────────────────────────────────┤
│  • Document Processor (Unstructured.io)                     │
│  • Text Cleaner                                             │
│  • Semantic Chunker                                         │
│  • Embedding Generator (Sentence-Transformers)             │
│  • Hybrid Retriever (Dense + BM25)                          │
│  • Cross-Encoder Reranker                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Storage & External Services Layer                          │
├─────────────────────────────────────────────────────────────┤
│  • ChromaDB (Vector Store)                                  │
│  • Ollama LLM Server                                        │
│  • File System (Document Upload)                            │
│  • Logging System                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Metadata Schema

```json
{
  "chunk": {
    "chunk_id": "550e8400-e29b-41d4-a716-446655440000_0",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "...",
    "embedding": "[0.123, -0.456, ...]",
    "metadata": {
      "file_name": "document.pdf",
      "source_type": "pdf",
      "page": 1,
      "section": "Introduction",
      "chunk_index": 0,
      "token_count": 450,
      "char_count": 2100,
      "embedding_model": "BAAI/bge-m3",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:05Z",
      "custom_metadata": {}
    }
  }
}
```

---

## Deployment Topology

```
┌──────────────────────────────────────────────────────────┐
│ Docker Compose Deployment                                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────┐  ┌───────────────────────┐ │
│  │  rag-api container     │  │  ollama container     │ │
│  │  ┌──────────────────┐  │  │  ┌─────────────────┐ │ │
│  │  │ FastAPI Server   │  │  │  │ Ollama LLM      │ │ │
│  │  │ Port: 8000       │  │  │  │ Port: 11434     │ │ │
│  │  │ Volumes:         │  │  │  │ Volume:         │ │ │
│  │  │ ├─ uploads/      │  │  │  │ └─ ollama_data/ │ │ │
│  │  │ ├─ chroma_db/    │  │  │  └─────────────────┘ │ │
│  │  │ └─ logs/         │  │  │                      │ │ │
│  │  └──────────────────┘  │  └───────────────────────┘ │
│  │                        │                            │ │
│  └────────────────────────┘  ┌───────────────────────┘ │
│            │                 │                         │
│            └────────rag-network─────────┘              │
│                                                        │
│  Volumes:                                             │
│  ├─ ollama_data (ollama models)                        │
│  └─ shared uploads/chroma_db/logs                      │
└──────────────────────────────────────────────────────────┘
```

---

## Scaling Architecture (Future)

```
┌────────────────────────────────────────────────────────────┐
│ Production Scaling Architecture                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐                                         │
│  │ Load Balancer│ (nginx/HAProxy)                         │
│  └────────┬─────┘                                         │
│           │                                               │
│     ┌─────┼─────┐                                         │
│     │     │     │                                         │
│     ▼     ▼     ▼                                         │
│  ┌────┐┌────┐┌────┐                                       │
│  │API1││API2││API3│  (Multiple API instances)            │
│  └─┬──┘└─┬──┘└─┬──┘                                       │
│    │     │     │                                         │
│    └─────┬─────┘                                         │
│          │                                               │
│    ┌─────▼────────┐                                      │
│    │ Message Queue│ (Redis/RabbitMQ)                     │
│    │ Background   │                                      │
│    │ Jobs         │                                      │
│    └─────┬────────┘                                      │
│          │                                               │
│    ┌─────▼──────────┐   ┌────────────┐   ┌───────────┐  │
│    │ ChromaDB       │   │ Embedding  │   │ Ollama    │  │
│    │ (Sharded)      │   │ Cache      │   │ Cluster   │  │
│    │                │   │ (Redis)    │   │           │  │
│    └────────────────┘   └────────────┘   └───────────┘  │
│                                                          │
│    ┌──────────────────────────────────────────────────┐ │
│    │ Monitoring & Logging                            │ │
│    │ ├─ Prometheus (metrics)                         │ │
│    │ ├─ ELK Stack (logs)                             │ │
│    │ └─ Grafana (dashboards)                         │ │
│    └──────────────────────────────────────────────────┘ │
│                                                         │
└────────────────────────────────────────────────────────────┘
```

---

## Performance Metrics

```
┌─────────────────────────────────────────────────────────┐
│ Key Performance Indicators                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Ingestion Pipeline:                                     │
│ • Document Processing: 2-5 min (10MB PDF)              │
│ • Embedding Generation: 50-100 ms/chunk                │
│ • Total Ingestion: 5-10 min (typical)                  │
│                                                         │
│ Retrieval Pipeline:                                     │
│ • Query Embedding: 10-50 ms                            │
│ • Dense Search: 10-50 ms                               │
│ • BM25 Search: 5-20 ms                                 │
│ • Reranking: 20-100 ms                                 │
│ • Total Retrieval: 50-200 ms                           │
│                                                         │
│ LLM Pipeline:                                           │
│ • Answer Generation: 1-3 sec                           │
│ • Token Generation: 20-50 tokens/sec                   │
│                                                         │
│ End-to-End Query:                                       │
│ • Complete RAG Query: 2-4 sec                          │
│ • P95 Response Time: 5-8 sec                           │
│                                                         │
│ Throughput:                                             │
│ • Single Instance: 5-10 queries/sec                    │
│ • Scalable to: 50-100 queries/sec (clustered)         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

```
User Request
    │
    ├─► Validation Error
    │   └─► 400 Bad Request
    │
    ├─► Authentication Error  
    │   └─► 401 Unauthorized
    │
    ├─► Processing Error
    │   ├─► Retry Logic
    │   └─► 500 Internal Error
    │
    ├─► External Service Error (Ollama/ChromaDB)
    │   ├─► Fallback Strategy
    │   └─► Circuit Breaker
    │
    └─► Success
        └─► 200 OK Response
```

---

## Security Considerations

```
┌────────────────────────────────────────────────────┐
│ Security Architecture                              │
├────────────────────────────────────────────────────┤
│                                                    │
│ Input Validation:                                  │
│ ├─ File type & size checks                        │
│ ├─ Content validation                             │
│ └─ Malware scanning (optional)                    │
│                                                    │
│ Data Protection:                                   │
│ ├─ Encryption at rest (optional)                  │
│ ├─ Encryption in transit (TLS)                    │
│ └─ Access control lists                           │
│                                                    │
│ API Security:                                      │
│ ├─ Rate limiting                                  │
│ ├─ CORS configuration                             │
│ ├─ API key authentication (optional)              │
│ └─ Request signing                                │
│                                                    │
│ Logging & Auditing:                               │
│ ├─ Request/response logging                       │
│ ├─ Error tracking                                 │
│ ├─ Access logs                                    │
│ └─ Compliance logging                             │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

**Architecture Version:** 1.0  
**Last Updated:** January 2024
