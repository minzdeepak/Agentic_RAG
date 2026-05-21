"""
Data models and schemas for RAG System
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Source document type"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"


class ChunkMetadata(BaseModel):
    """Metadata for a chunk"""
    document_id: str
    file_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str
    source_type: SourceType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    token_count: int
    char_count: int
    embedding_model: str
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """A chunk of a document with embedding"""
    chunk_id: str
    document_id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None


class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion"""
    file_name: str
    source_type: SourceType
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion"""
    document_id: str
    file_name: str
    chunks_created: int
    status: str
    message: str
    created_at: datetime


class RAGQuery(BaseModel):
    """Query model for RAG system"""
    question: str
    top_k: int = Field(default=3, ge=1, le=10)
    include_sources: bool = True
    use_hybrid_search: bool = True


class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float
    processing_time: float
    model: str
    llm_model: str


class HealthStatus(BaseModel):
    """System health status"""
    status: str
    vectordb: str
    embeddings: str
    llm: str
    timestamp: datetime


class RetrievalResult(BaseModel):
    """Result from retrieval"""
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
