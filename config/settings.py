"""
Configuration and Settings for RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# API Configuration
API_TITLE = "RAG System API"
API_VERSION = "1.0.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # cuda or cpu

# Chunking Configuration
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 800
OVERLAP = 50  # Token overlap between chunks
SIMILARITY_THRESHOLD = 0.7

# Vector Database Configuration
VECTOR_DB_TYPE = "chromadb"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DB_DIR)

# Retrieval Configuration
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", 5))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 5))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 3))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | gemini | openai | claude

# LLM Configuration (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.6.79:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", 0.9))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 60))

# LLM Configuration (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

# LLM Configuration (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# LLM Configuration (Claude)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

# File Upload Configuration
ALLOWED_FILE_TYPES = {"pdf", "docx", "doc", "txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Processing Configuration
BATCH_SIZE = 10
ENABLE_METRICS = True
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Metadata Configuration
STORE_CHUNK_EMBEDDINGS = True
ENABLE_LINEAGE_TRACKING = True
