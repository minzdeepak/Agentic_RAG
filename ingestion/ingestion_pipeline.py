"""
Ingestion pipeline for processing and storing documents
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
from datetime import datetime

from ingestion.document_processor import DocumentProcessor
from ingestion.data_cleaner import DataCleaner
from chunking.semantic_chunker import SemanticChunker
from embeddings.embedding_model import EmbeddingModel
from vectordb.chromadb_store import ChromaVectorStore
from utils.logger import get_logger
from utils.helpers import generate_document_id, generate_chunk_id, count_tokens_approximate
from models.schemas import SourceType

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Complete ingestion pipeline:
    Document Upload → Processing → Cleaning → Chunking → 
    Embedding → Vector Storage
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore
    ):
        """Initialize ingestion pipeline"""
        self.logger = logger
        self.document_processor = DocumentProcessor()
        self.data_cleaner = DataCleaner()
        self.semantic_chunker = SemanticChunker(embedding_model)
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def ingest_document(
        self,
        file_path: str,
        source_type: SourceType,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete ingestion workflow for a single document
        
        Args:
            file_path: Path to the document
            source_type: Type of document (pdf, docx, etc.)
            document_metadata: Additional metadata
            
        Returns:
            Ingestion result with document_id and chunks created
        """
        self.logger.info(f"Starting ingestion for: {file_path}")
        
        try:
            # Step 1: Process document
            self.logger.info("Step 1: Processing document...")
            raw_text, structured_elements = self.document_processor.process_document(file_path)
            
            # Step 2: Clean data
            self.logger.info("Step 2: Cleaning data...")
            cleaned_text = self.data_cleaner.clean_text(raw_text)
            cleaned_text = self.data_cleaner.remove_boilerplate(cleaned_text)
            
            # Step 3: Semantic chunking
            self.logger.info("Step 3: Semantic chunking...")
            chunks = self.semantic_chunker.chunk_document(cleaned_text, structured_elements)
            
            # Step 4: Generate embeddings and store
            self.logger.info("Step 4: Generating embeddings and storing...")
            document_id = generate_document_id()
            stored_chunks = self._store_chunks(
                document_id=document_id,
                file_path=file_path,
                source_type=source_type,
                chunks=chunks,
                metadata=document_metadata
            )
            
            result = {
                'document_id': document_id,
                'file_name': Path(file_path).name,
                'chunks_created': len(stored_chunks),
                'total_tokens': sum(chunk['token_count'] for chunk in stored_chunks),
                'status': 'success',
                'message': f'Successfully ingested {len(stored_chunks)} chunks',
                'created_at': datetime.utcnow().isoformat(),
                'source_type': source_type
            }
            
            self.logger.info(f"Ingestion complete. Document ID: {document_id}, Chunks: {len(stored_chunks)}")
            return result
        
        except Exception as e:
            self.logger.error(f"Ingestion failed: {str(e)}")
            raise
    
    def _store_chunks(
        self,
        document_id: str,
        file_path: str,
        source_type: SourceType,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Store chunks in vector database
        
        Args:
            document_id: Document ID
            file_path: Original file path
            source_type: Document source type
            chunks: List of chunks
            metadata: Additional metadata
            
        Returns:
            List of stored chunks with metadata
        """
        file_name = Path(file_path).name
        
        chunk_texts = []
        chunk_embeddings = []
        chunk_metadatas = []
        chunk_ids = []
        stored_chunks = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(document_id, idx)
            chunk_text = chunk['content']
            
            # Generate embedding
            try:
                embedding = self.embedding_model.embed(chunk_text)
            except Exception as e:
                self.logger.warning(f"Failed to embed chunk {chunk_id}: {str(e)}")
                continue
            
            # Build metadata
            chunk_metadata = {
                'document_id': document_id,
                'file_name': file_name,
                'chunk_id': chunk_id,
                'chunk_index': idx,
                'source_type': source_type,
                'token_count': chunk['token_count'],
                'char_count': chunk['char_count'],
                'created_at': datetime.utcnow().isoformat(),
            }
            
            # Add optional metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            chunk_texts.append(chunk_text)
            chunk_embeddings.append(embedding)
            chunk_metadatas.append(chunk_metadata)
            chunk_ids.append(chunk_id)
            
            stored_chunks.append({
                **chunk,
                'chunk_id': chunk_id,
                'metadata': chunk_metadata
            })
        
        # Store in vector database
        if chunk_texts:
            try:
                self.vector_store.add_documents(
                    documents=chunk_texts,
                    embeddings=chunk_embeddings,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
                self.logger.info(f"Stored {len(chunk_texts)} chunks in vector database")
            except Exception as e:
                self.logger.error(f"Error storing chunks: {str(e)}")
                raise
        
        return stored_chunks
    
    def ingest_batch(
        self,
        file_paths: List[Tuple[str, SourceType]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents
        
        Args:
            file_paths: List of (file_path, source_type) tuples
            metadata: Optional metadata for all documents
            
        Returns:
            List of ingestion results
        """
        results = []
        
        for file_path, source_type in file_paths:
            try:
                result = self.ingest_document(file_path, source_type, metadata)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to ingest {file_path}: {str(e)}")
                results.append({
                    'file_name': Path(file_path).name,
                    'status': 'failed',
                    'message': str(e)
                })
        
        return results
