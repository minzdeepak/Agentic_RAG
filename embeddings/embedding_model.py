"""
Embedding module using BAAI/bge-m3
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from utils.logger import get_logger
from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE
)

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize embedding model"""
        self.logger = logger
        self.model_name = model_name
        
        # Determine device
        if EMBEDDING_DEVICE == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        self.logger.info(f"Loading embedding model: {model_name} on device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = EMBEDDING_BATCH_SIZE) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Embedding(s) as numpy array(s)
        """
        # Handle single text
        if isinstance(texts, str):
            try:
                embedding = self.model.encode(texts, convert_to_numpy=True)
                return embedding
            except Exception as e:
                self.logger.error(f"Error embedding text: {str(e)}")
                raise
        
        # Handle batch of texts
        if isinstance(texts, list):
            try:
                self.logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                return embeddings
            except Exception as e:
                self.logger.error(f"Error embedding batch: {str(e)}")
                raise
    
    def embed_texts_with_instruction(
        self,
        texts: List[str],
        instruction: str = "Represent this document for retrieval:",
        batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        """
        Embed texts with instruction prefix (useful for retrieval tasks)
        BGE models perform better with instruction prefix
        
        Args:
            texts: List of texts to embed
            instruction: Instruction prefix to add
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        # Add instruction prefix
        prefixed_texts = [f"{instruction} {text}" if text else text for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query with optimized instruction for search
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        # BGE models use specific instruction for queries
        instruction = "Represent this query for retrieval:"
        prefixed_query = f"{instruction} {query}"
        
        embedding = self.model.encode(prefixed_query, convert_to_numpy=True)
        return embedding
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = float(np.dot(vec1_norm, vec2_norm))
        return similarity
    
    def batch_cosine_similarities(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and multiple embeddings efficiently
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embeddings (n_samples, embedding_dim)
            
        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.get_max_seq_length() if hasattr(self.model, 'get_max_seq_length') else "Unknown"
        }
