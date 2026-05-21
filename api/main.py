"""
FastAPI application for RAG System
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List
import shutil

from models.schemas import (
    DocumentIngestionRequest, DocumentIngestionResponse,
    RAGQuery, RAGResponse, HealthStatus, SourceType
)
from ingestion.ingestion_pipeline import IngestionPipeline
from embeddings.embedding_model import EmbeddingModel
from vectordb.chromadb_store import ChromaVectorStore
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_factory import get_llm
from utils.logger import get_logger
from config.settings import (
    API_TITLE, API_VERSION, ALLOWED_FILE_TYPES, MAX_FILE_SIZE,
    UPLOAD_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, RERANKER_MODEL
)
import time

logger = get_logger(__name__)

# Initialize FastAPI
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Production-Grade RAG System"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
embedding_model: EmbeddingModel = None
vector_store: ChromaVectorStore = None
ingestion_pipeline: IngestionPipeline = None
hybrid_retriever: HybridRetriever = None
llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global embedding_model, vector_store, ingestion_pipeline, hybrid_retriever, llm
    
    logger.info("Initializing RAG System...")
    
    try:
        # Initialize embedding model
        logger.info("Loading embedding model...")
        embedding_model = EmbeddingModel(EMBEDDING_MODEL)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = ChromaVectorStore(CHROMA_COLLECTION_NAME)
        
        # Initialize ingestion pipeline
        logger.info("Setting up ingestion pipeline...")
        ingestion_pipeline = IngestionPipeline(embedding_model, vector_store)
        
        # Initialize retriever
        logger.info("Setting up hybrid retriever...")
        hybrid_retriever = HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            reranker_model=RERANKER_MODEL
        )
        
        # Initialize LLM
        logger.info("Initializing LLM provider...")
        llm = get_llm()
        
        logger.info("RAG System initialized successfully!")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG System: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG System...")
    if vector_store:
        vector_store.persist()


# ==================== API Endpoints ====================

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Check system health"""
    return HealthStatus(
        status="healthy",
        vectordb="ready" if vector_store else "not_initialized",
        embeddings="ready" if embedding_model else "not_initialized",
        llm="ready" if llm else "not_initialized",
        timestamp=__import__('datetime').datetime.utcnow()
    )


@app.post("/api/v1/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_metadata: str = None
):
    """
    Ingest a document (PDF, DOCX, DOC, TXT)
    
    - **file**: Document file to ingest
    - **document_metadata**: Optional JSON metadata
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lstrip('.').lower()
        if file_ext not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type '.{file_ext}' not allowed. Allowed: {ALLOWED_FILE_TYPES}"
            )
        
        # Validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds limit of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Save file temporarily
        temp_file_path = UPLOAD_DIR / file.filename
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        
        # Determine source type
        source_type = SourceType[file_ext.upper()] if file_ext.upper() in SourceType.__members__ else SourceType.TXT
        
        # Parse metadata if provided
        metadata = None
        if document_metadata:
            import json
            try:
                metadata = json.loads(document_metadata)
            except:
                logger.warning("Failed to parse metadata JSON")
        
        # Ingest document
        logger.info(f"Ingesting document: {file.filename}")
        result = ingestion_pipeline.ingest_document(
            file_path=str(temp_file_path),
            source_type=source_type,
            document_metadata=metadata
        )
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        return DocumentIngestionResponse(
            document_id=result['document_id'],
            file_name=result['file_name'],
            chunks_created=result['chunks_created'],
            status='success',
            message=result['message'],
            created_at=__import__('datetime').datetime.fromisoformat(result['created_at'])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=RAGResponse)
async def query_rag(query_request: RAGQuery):
    """
    Query the RAG system
    
    - **question**: The question to answer
    - **top_k**: Number of context documents to retrieve
    - **include_sources**: Whether to include source information
    - **use_hybrid_search**: Whether to use hybrid search (dense + BM25)
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing query: {query_request.question}")
        
        # Step 1: Retrieve relevant documents
        logger.info("Step 1: Retrieving documents...")
        retrieved_docs = hybrid_retriever.retrieve(
            query=query_request.question,
            top_k=query_request.top_k,
            use_hybrid=query_request.use_hybrid_search
        )
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=400,
                detail="No relevant documents found for this query"
            )
        
        # Step 2: Extract context
        context = [doc['content'] for doc in retrieved_docs]
        
        # Step 3: Generate answer
        logger.info("Step 2: Generating answer with LLM...")
        answer_response = llm.generate_answer(
            query=query_request.question,
            context=context
        )
        
        # Step 4: Calculate confidence
        if retrieved_docs:
            avg_score = sum(doc.get('combined_score', doc.get('score', 0)) for doc in retrieved_docs) / len(retrieved_docs)
            confidence = min(avg_score, 1.0)
        else:
            confidence = 0.0
        
        # Step 5: Prepare sources
        sources = []
        if query_request.include_sources:
            for doc in retrieved_docs:
                sources.append({
                    'content': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'score': doc.get('combined_score', doc.get('score', 0)),
                    'metadata': doc.get('metadata', {})
                })
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            question=query_request.question,
            answer=answer_response['answer'],
            sources=sources,
            confidence=float(confidence),
            processing_time=processing_time,
            model=embedding_model.model_name,
            llm_model=llm.model
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/vectordb/info")
async def get_vectordb_info():
    """Get vector database information"""
    try:
        info = vector_store.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting vector DB info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def get_models_info():
    """Get information about loaded models"""
    try:
        embedding_info = embedding_model.get_model_info()
        llm_info = llm.get_model_info()

        return {
            'embedding': embedding_info,
            'llm': llm_info,
            'available_llm_models': llm.list_available_models()
        }
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/vectordb/clear")
async def clear_vectordb():
    """Clear vector database (use with caution!)"""
    global vector_store
    try:
        vector_store.delete_collection()
        # Recreate collection
        vector_store = ChromaVectorStore(CHROMA_COLLECTION_NAME)
        
        return {"status": "success", "message": "Vector database cleared"}
    except Exception as e:
        logger.error(f"Error clearing vector DB: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    from config.settings import API_HOST, API_PORT
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
