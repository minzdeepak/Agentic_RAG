"""
Retrieval module with hybrid search (dense + BM25) and reranking
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict

from utils.logger import get_logger
from embeddings.embedding_model import EmbeddingModel
from vectordb.chromadb_store import ChromaVectorStore
from config.settings import TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining dense vector search and BM25 keyword search
    
    Retrieval Architecture:
    Question → Embedding Generation → Dense Vector Search → 
    BM25 Keyword Search → Merge Results → Reranker → 
    Top Context Selection → LLM Answer Generation
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        reranker_model: str = None
    ):
        """Initialize retriever"""
        self.logger = logger
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker_model = reranker_model
        self.bm25_index = None
        self.documents_cache = {}
        
        # Initialize reranker if specified
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
                self.logger.info(f"Reranker loaded: {reranker_model}")
            except:
                self.logger.warning("Could not load reranker model")
                self.reranker = None
        else:
            self.reranker = None
    
    def build_bm25_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
        """
        self.logger.info(f"Building BM25 index for {len(documents)} documents")
        
        try:
            # Tokenize documents
            tokenized_docs = [doc.split() for doc in documents]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.documents_cache = {i: doc for i, doc in enumerate(documents)}
            
            self.logger.info("BM25 index built successfully")
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {str(e)}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        use_hybrid: bool = True,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Query text
            top_k: Number of final results
            use_hybrid: Whether to use hybrid search
            alpha: Weight for combining dense and BM25 scores (0-1)
            
        Returns:
            List of retrieved documents with scores
        """
        self.logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        if not use_hybrid or self.bm25_index is None:
            # Use only dense search
            return self._dense_search(query, top_k)
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 2: Dense vector search
        dense_results = self._dense_search_with_embedding(query_embedding, top_k=TOP_K_DENSE)
        
        # Step 3: BM25 keyword search
        bm25_results = self._bm25_search(query, top_k=TOP_K_BM25)
        
        # Step 4: Merge results
        merged_results = self._merge_results(dense_results, bm25_results, alpha)
        
        # Step 5: Rerank if available
        if self.reranker:
            merged_results = self._rerank_results(query, merged_results)
        
        # Step 6: Select top K results
        final_results = merged_results[:top_k]
        
        self.logger.info(f"Retrieved {len(final_results)} relevant documents")
        return final_results
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure dense vector search"""
        query_embedding = self.embedding_model.embed_query(query)
        return self._dense_search_with_embedding(query_embedding, top_k)
    
    def _dense_search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_DENSE
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search using ChromaDB
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to retrieve
            
        Returns:
            List of search results
        """
        try:
            results = self.vector_store.search(query_embedding, n_results=top_k)
            
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            ids = results.get('ids', [[]])[0]
            
            # Convert distances to similarity scores (cosine distance -> similarity)
            scores = [1 - dist for dist in distances]
            
            retrieved = []
            for i, (doc, metadata, score, doc_id) in enumerate(zip(documents, metadatas, scores, ids)):
                retrieved.append({
                    'id': doc_id,
                    'content': doc,
                    'metadata': metadata,
                    'score': float(score),
                    'rank': i + 1,
                    'search_type': 'dense'
                })
            
            return retrieved
        
        except Exception as e:
            self.logger.error(f"Error in dense search: {str(e)}")
            return []
    
    def _bm25_search(self, query: str, top_k: int = TOP_K_BM25) -> List[Dict[str, Any]]:
        """
        BM25 keyword search
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of BM25 search results
        """
        if not self.bm25_index or not self.documents_cache:
            self.logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top K results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            retrieved = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include positive scores
                    retrieved.append({
                        'id': f"bm25_{idx}",
                        'content': self.documents_cache.get(idx, ''),
                        'metadata': {},
                        'score': float(scores[idx]),
                        'rank': rank + 1,
                        'search_type': 'bm25',
                        'doc_index': idx
                    })
            
            return retrieved
        
        except Exception as e:
            self.logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def _merge_results(
        self,
        dense_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and BM25 results with scoring
        
        Args:
            dense_results: Dense search results
            bm25_results: BM25 search results
            alpha: Weight for dense search (1-alpha for BM25)
            
        Returns:
            Merged and ranked results
        """
        # Normalize scores to 0-1 range
        merged_scores = defaultdict(lambda: {'dense': 0, 'bm25': 0, 'data': None})
        
        # Add dense scores
        max_dense_score = max([r['score'] for r in dense_results]) if dense_results else 1
        for result in dense_results:
            normalized_score = result['score'] / max_dense_score if max_dense_score > 0 else 0
            content_id = result['content'][:100]  # Use content as key
            merged_scores[content_id]['dense'] = normalized_score
            merged_scores[content_id]['data'] = result
        
        # Add BM25 scores
        max_bm25_score = max([r['score'] for r in bm25_results]) if bm25_results else 1
        for result in bm25_results:
            normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
            content_id = result['content'][:100]
            merged_scores[content_id]['bm25'] = normalized_score
            if merged_scores[content_id]['data'] is None:
                merged_scores[content_id]['data'] = result
        
        # Calculate combined scores
        final_results = []
        for content_id, scores_dict in merged_scores.items():
            combined_score = (
                alpha * scores_dict['dense'] +
                (1 - alpha) * scores_dict['bm25']
            )
            
            result = scores_dict['data'].copy()
            result['combined_score'] = combined_score
            result['dense_score'] = scores_dict['dense']
            result['bm25_score'] = scores_dict['bm25']
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = TOP_K_FINAL
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Query text
            results: Search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not self.reranker or not results:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, result['content']] for result in results]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by rerank score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return results[:top_k]
        
        except Exception as e:
            self.logger.warning(f"Reranking failed, using original order: {str(e)}")
            return results[:top_k]
