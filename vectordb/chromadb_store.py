"""
Vector Database abstraction using ChromaDB
"""
from typing import List, Dict, Optional, Any
import chromadb
import uuid
from pathlib import Path

from utils.logger import get_logger
from config.settings import (
    CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, STORE_CHUNK_EMBEDDINGS
)

logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB vector storage and retrieval"""
    
    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME):
        """
        Initialize ChromaDB client
        
        Args:
            collection_name: Name of the collection to use
        """
        self.logger = logger
        self.collection_name = collection_name
        
        self.logger.info(f"Initializing ChromaDB with collection: {collection_name}")
        
        try:
            # Ensure persist directory exists
            Path(CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client using new API
            self.client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info(f"Collection '{collection_name}' ready")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str] = None
    ) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of document texts
            embeddings: List of embeddings
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        self.logger.info(f"Adding {len(documents)} documents to collection")
        
        try:
            # Convert embeddings to list format if needed
            embeddings_list = [
                emb.tolist() if hasattr(emb, 'tolist') else emb
                for emb in embeddings
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully added {len(documents)} documents")
            return ids
        
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with results
        """
        try:
            # Convert embedding to list if needed
            query_emb = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                where=where,
                include=["embeddings", "metadatas", "documents", "distances"]
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by ID
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None
        """
        try:
            results = self.collection.get(ids=[doc_id], include=["embeddings", "metadatas", "documents"])
            
            if results['ids']:
                return {
                    "id": results['ids'][0],
                    "document": results['documents'][0],
                    "metadata": results['metadatas'][0],
                    "embedding": results['embeddings'][0] if STORE_CHUNK_EMBEDDINGS else None
                }
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting document: {str(e)}")
            return None
    
    def update_documents(
        self,
        ids: List[str],
        documents: List[str] = None,
        embeddings: List[List[float]] = None,
        metadatas: List[Dict[str, Any]] = None
    ) -> None:
        """
        Update existing documents
        
        Args:
            ids: Document IDs to update
            documents: Updated documents
            embeddings: Updated embeddings
            metadatas: Updated metadata
        """
        try:
            update_data = {"ids": ids}
            
            if documents:
                update_data["documents"] = documents
            
            if embeddings:
                embeddings_list = [
                    emb.tolist() if hasattr(emb, 'tolist') else emb
                    for emb in embeddings
                ]
                update_data["embeddings"] = embeddings_list
            
            if metadatas:
                update_data["metadatas"] = metadatas
            
            self.collection.update(**update_data)
            self.logger.info(f"Updated {len(ids)} documents")
        
        except Exception as e:
            self.logger.error(f"Error updating documents: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def persist(self) -> None:
        """Persist data to disk"""
        try:
            self.client.persist()
            self.logger.info("ChromaDB persisted to disk")
        except Exception as e:
            self.logger.warning(f"Error persisting ChromaDB: {str(e)}")
    
    def delete_collection(self) -> None:
        """Delete entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
