# 🚀 Complete Deployment & Configuration Guide

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Installation Methods](#installation-methods)
3. [Docker Deployment](#docker-deployment)
4. [Configuration Guide](#configuration-guide)
5. [Initial Setup](#initial-setup)
6. [Testing & Validation](#testing--validation)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Pre-Deployment Checklist

### System Requirements

```
✓ Python 3.11 or higher
✓ 8GB RAM minimum (16GB recommended)
✓ 4GB disk space for models & data
✓ GPU with CUDA 11.8+ (optional, for acceleration)
✓ Network access to Ollama server
✓ Docker & Docker Compose (for containerized deployment)
```

### Required Software

```bash
# Check Python version
python --version  # Should be 3.11+

# Verify Ollama installation
ollama --version

# Check Docker (if using Docker deployment)
docker --version
docker-compose --version
```

---

## Installation Methods

### Method 1: Local Installation (Recommended for Development)

#### Step 1: Clone and Navigate

```bash
cd rag-system
```

#### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

Key settings to verify:
```env
OLLAMA_BASE_URL=http://192.168.6.79:11434
EMBEDDING_DEVICE=cuda  # or cpu
LOG_LEVEL=INFO
```

#### Step 5: Initialize Ollama

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull the model
ollama pull llama3.1

# Terminal 3: Verify installation
curl http://localhost:11434/api/tags
```

#### Step 6: Run Application

```bash
# Option A: Direct Python
python main.py

# Option B: Using start script
# Windows:
start.bat
# macOS/Linux:
chmod +x start.sh
./start.sh
```

The application will be available at `http://localhost:8000`

---

### Method 2: Docker Deployment (Recommended for Production)

#### Step 1: Build and Start Services

```bash
cd rag-system
docker-compose up --build
```

#### Step 2: Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Ollama
curl http://localhost:11434/api/tags

# View logs
docker logs rag-system-api
docker logs ollama-server
```

#### Step 3: Test API

```bash
curl -X GET "http://localhost:8000/health"
```

---

## Docker Deployment

### Understanding docker-compose.yml

```yaml
version: '3.8'

services:
  rag-api:
    # Build from Dockerfile in current directory
    build:
      context: .
      dockerfile: Dockerfile
    
    # Container name for easy reference
    container_name: rag-system-api
    
    # Port mapping: host:container
    ports:
      - "8000:8000"
    
    # Environment variables
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - EMBEDDING_DEVICE=cpu  # Change to 'cuda' for GPU
      - OLLAMA_BASE_URL=http://ollama:11434
    
    # Volume mounts for persistence
    volumes:
      - ./uploads:/app/uploads
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    
    # Dependencies
    depends_on:
      - ollama
    
    # Network
    networks:
      - rag-network
    
    # Restart policy
    restart: unless-stopped
```

### Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f rag-api
docker-compose logs -f ollama

# Execute commands
docker-compose exec rag-api python test_rag_system.py

# Scale services
docker-compose up -d --scale rag-api=3

# Remove volumes (careful!)
docker-compose down -v
```

### GPU Support with Docker

If you have NVIDIA GPU:

```yaml
services:
  rag-api:
    runtime: nvidia
    environment:
      - EMBEDDING_DEVICE=cuda
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /usr/local/cuda:/usr/local/cuda
```

---

## Configuration Guide

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

### Configuration Parameters

#### API Configuration

```env
# Server binding
API_HOST=0.0.0.0           # Listen on all interfaces
API_PORT=8000              # Port number

# Security (optional)
API_KEY_ENABLED=false
API_KEY=your-secret-key
```

#### Embedding Configuration

```env
# Model selection
EMBEDDING_MODEL=BAAI/bge-m3

# Hardware acceleration
EMBEDDING_DEVICE=cuda      # cuda or cpu
EMBEDDING_BATCH_SIZE=32    # Adjust based on GPU memory

# If OOM errors:
# EMBEDDING_BATCH_SIZE=8
# EMBEDDING_DEVICE=cpu
```

#### Vector Database Configuration

```env
CHROMA_COLLECTION_NAME=rag_documents

# Advanced (edit config/settings.py):
MIN_CHUNK_SIZE=300         # Minimum tokens
MAX_CHUNK_SIZE=800         # Maximum tokens
OVERLAP=50                 # Token overlap
SIMILARITY_THRESHOLD=0.7   # Semantic boundary detection
```

#### Retrieval Configuration

```env
# Search parameters
TOP_K_DENSE=5              # Dense search results
TOP_K_BM25=5               # BM25 search results
TOP_K_FINAL=3              # Final results

# Reranking
RERANKER_MODEL=BAAI/bge-reranker-large
# Set to "" or comment out to disable reranking
```

#### LLM Configuration

```env
# Ollama server
OLLAMA_BASE_URL=http://192.168.6.79:11434
# For Docker: http://ollama:11434

# Model and parameters
OLLAMA_MODEL=llama3.1
OLLAMA_TEMPERATURE=0.7     # 0-1: Lower = more deterministic
OLLAMA_TOP_P=0.9           # 0-1: Lower = more focused
OLLAMA_TIMEOUT=60          # Seconds
```

#### Logging Configuration

```env
LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR
# Logs stored in: logs/
```

### Advanced Configuration (config/settings.py)

Edit `config/settings.py` for fine-tuning:

```python
# Chunking
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 800
OVERLAP = 50
SIMILARITY_THRESHOLD = 0.7

# File upload
ALLOWED_FILE_TYPES = {"pdf", "docx", "doc", "txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Processing
BATCH_SIZE = 10
ENABLE_METRICS = True
STORE_CHUNK_EMBEDDINGS = True
ENABLE_LINEAGE_TRACKING = True
```

---

## Initial Setup

### Step 1: Verify Prerequisites

```bash
# Python version
python --version

# Pip packages
pip list | grep -E "fastapi|torch|chromadb"

# Ollama
ollama list

# Network connectivity
ping 192.168.6.79  # Or your Ollama server
```

### Step 2: Test Components

```bash
# Run quick test
python QUICKSTART.md

# Run unit tests
pytest test_rag_system.py -v

# Run examples
python examples.py
```

### Step 3: Initialize Vector Database

```bash
python -c "
from vectordb.chromadb_store import ChromaVectorStore
vs = ChromaVectorStore()
info = vs.get_collection_info()
print(f'Vector DB Status: OK')
print(f'Documents: {info[\"document_count\"]}')
"
```

---

## Testing & Validation

### API Endpoint Testing

```bash
# Health check
curl http://localhost:8000/health

# Get models info
curl http://localhost:8000/api/v1/models

# Vector DB info
curl http://localhost:8000/api/v1/vectordb/info

# Ingest test document
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@test_document.pdf"

# Query test
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "top_k": 3,
    "include_sources": true,
    "use_hybrid_search": true
  }'
```

### Unit Tests

```bash
# Run all tests
pytest test_rag_system.py -v

# Run specific test
pytest test_rag_system.py::TestSemanticChunker::test_chunk_document -v

# With coverage
pytest test_rag_system.py --cov=. --cov-report=html
```

### Integration Tests

```bash
# Run example scripts
python examples.py
# Choose option 6 (Health Check) to verify all components
```

### Performance Testing

```bash
# Load test (using Apache Bench)
ab -n 100 -c 10 http://localhost:8000/health

# Or use Apache JMeter, Locust, etc.
```

---

## Production Deployment

### Pre-Production Checklist

```
Security:
☐ Set LOG_LEVEL=WARNING (not DEBUG)
☐ Disable CORS if not needed
☐ Set up API authentication
☐ Enable HTTPS/TLS
☐ Use environment variables for secrets
☐ Regular security updates

Performance:
☐ Use GPU if available (EMBEDDING_DEVICE=cuda)
☐ Increase EMBEDDING_BATCH_SIZE
☐ Set appropriate TOP_K values
☐ Enable caching if needed
☐ Monitor memory usage

Reliability:
☐ Set up health checks
☐ Configure restart policies
☐ Implement error handling
☐ Set up monitoring/alerting
☐ Regular backups of vector DB

Scalability:
☐ Load balancer configuration
☐ Horizontal scaling setup
☐ Database optimization
☐ Cache layer (Redis)
☐ Message queue for async jobs
```

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    
    environment:
      - LOG_LEVEL=WARNING
      - EMBEDDING_DEVICE=cuda
      - API_HOST=0.0.0.0
      - API_PORT=8000
    
    volumes:
      - shared_uploads:/app/uploads
      - shared_chroma:/app/chroma_db
    
    restart: always
    networks:
      - prod-network

  load-balancer:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - prod-network

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - prod-network

volumes:
  shared_uploads:
  shared_chroma:
  ollama_data:

networks:
  prod-network:
```

### Kubernetes Deployment

For Kubernetes, create:
- `deployment.yaml` - API deployment
- `service.yaml` - API service
- `pvc.yaml` - Persistent volumes
- `configmap.yaml` - Configuration

---

## Monitoring & Maintenance

### Logging

```bash
# View logs in real-time
tail -f logs/api.log

# Check for errors
grep ERROR logs/*.log

# Log rotation (configure in production)
# Use logrotate or similar tool
```

### Health Monitoring

```bash
# Regular health checks
watch -n 5 'curl -s http://localhost:8000/health | jq .'

# Monitor vector DB
watch -n 10 'curl -s http://localhost:8000/api/v1/vectordb/info | jq .'

# System resources
watch -n 5 'docker stats'
```

### Maintenance Tasks

```bash
# Backup vector database
tar czf backup_chroma_$(date +%Y%m%d).tar.gz chroma_db/

# Clear old logs
find logs -mtime +30 -delete

# Restart services
docker-compose restart rag-api

# Update dependencies (test first!)
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

### Performance Optimization

```python
# Monitor query times
from utils.logger import get_logger
logger = get_logger("performance")

import time
start = time.time()
# ... perform operation ...
duration = time.time() - start
logger.info(f"Operation took {duration:.2f}s")

# Identify slow operations
# Adjust parameters in config/settings.py
```

---

## Troubleshooting Production Issues

### High Memory Usage

```bash
# Check memory
free -h
docker stats

# Solutions:
# 1. Reduce EMBEDDING_BATCH_SIZE
# 2. Switch to CPU (EMBEDDING_DEVICE=cpu)
# 3. Clear old chunks: DELETE /api/v1/vectordb/clear
# 4. Increase server RAM
```

### Slow Queries

```python
# Check query time
# Adjust TOP_K values (decrease for speed, increase for quality)
TOP_K_DENSE=3
TOP_K_BM25=3
TOP_K_FINAL=2

# Disable reranking if too slow
RERANKER_MODEL=""
```

### Connection Errors

```bash
# Test Ollama connectivity
curl http://ollama:11434/api/tags

# If using remote Ollama:
# Update OLLAMA_BASE_URL=http://remote-ip:11434

# Test ChromaDB
docker exec rag-system-api python -c "from vectordb.chromadb_store import ChromaVectorStore; ChromaVectorStore()"
```

### Vector DB Issues

```bash
# Rebuild index
docker exec rag-system-api python -c "
from vectordb.chromadb_store import ChromaVectorStore
vs = ChromaVectorStore()
vs.delete_collection()
# Re-ingest documents
"
```

---

## Backup & Recovery

### Regular Backups

```bash
#!/bin/bash
# Daily backup script

BACKUP_DIR="/backups/rag-system"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup vector database
tar czf "$BACKUP_DIR/chroma_$TIMESTAMP.tar.gz" chroma_db/

# Backup uploads
tar czf "$BACKUP_DIR/uploads_$TIMESTAMP.tar.gz" uploads/

# Backup logs
tar czf "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" logs/

# Keep only last 30 days
find $BACKUP_DIR -mtime +30 -delete
```

### Recovery

```bash
# Restore vector database
tar xzf backup_chroma_20240115.tar.gz -C /

# Restore uploads
tar xzf backup_uploads_20240115.tar.gz -C /

# Restart services
docker-compose restart rag-api
```

---

## Support & Resources

- **Documentation:** README.md, ARCHITECTURE.md
- **Examples:** examples.py
- **Tests:** test_rag_system.py
- **Quick Start:** QUICKSTART.md

---

**Last Updated:** January 2024  
**Version:** 1.0.0
