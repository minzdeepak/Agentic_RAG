"""
Unit tests for RAG System components
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Test configuration
@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Chapter 1: Introduction
    
    This is the introduction. Here are key points:
    - Point 1
    - Point 2
    
    Chapter 2: Main Content
    
    The main content goes here.
    """


@pytest.fixture
def sample_embedding():
    """Sample embedding vector"""
    return np.random.randn(1024)


# Tests for Semantic Chunker
class TestSemanticChunker:
    
    def test_chunk_document(self, sample_text):
        """Test document chunking"""
        from chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_text)
        
        assert len(chunks) > 0
        assert all('chunk_index' in c for c in chunks)
        assert all('content' in c for c in chunks)
        assert all('token_count' in c for c in chunks)
    
    def test_structural_parsing(self, sample_text):
        """Test structural parsing"""
        from chunking.semantic_chunker import SemanticChunker
        
        chunker = SemanticChunker()
        segments = chunker._structural_parsing(sample_text)
        
        assert len(segments) > 0
        assert any(s.segment_type == 'heading' for s in segments)


# Tests for Embeddings
class TestEmbeddings:
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_text(self, mock_model):
        """Test text embedding"""
        from embeddings.embedding_model import EmbeddingModel
        
        mock_model.return_value.encode.return_value = np.random.randn(1024)
        mock_model.return_value.get_sentence_embedding_dimension.return_value = 1024
        
        model = EmbeddingModel()
        embedding = model.embed("Test text")
        
        assert embedding.shape == (1024,)
    
    def test_cosine_similarity(self, sample_embedding):
        """Test cosine similarity calculation"""
        from embeddings.embedding_model import EmbeddingModel
        
        model = EmbeddingModel.__new__(EmbeddingModel)
        
        similarity = model.cosine_similarity(sample_embedding, sample_embedding)
        assert abs(similarity - 1.0) < 0.001  # Should be exactly 1.0


# Tests for Data Cleaning
class TestDataCleaner:
    
    def test_clean_text(self):
        """Test text cleaning"""
        from ingestion.data_cleaner import DataCleaner
        
        cleaner = DataCleaner()
        dirty_text = "Hello   world\n\n\nvisit http://example.com for info"
        
        cleaned = cleaner.clean_text(dirty_text)
        
        assert "http://example.com" not in cleaned
        assert cleaned != dirty_text
        assert "Hello" in cleaned
    
    def test_remove_boilerplate(self):
        """Test boilerplate removal"""
        from ingestion.data_cleaner import DataCleaner
        
        cleaner = DataCleaner()
        text = "Important content\nCopyright © 2024\nMore content"
        
        cleaned = cleaner.remove_boilerplate(text)
        
        assert "Copyright" not in cleaned
        assert "Important" in cleaned


# Tests for Helpers
class TestHelpers:
    
    def test_generate_ids(self):
        """Test ID generation"""
        from utils.helpers import generate_document_id, generate_chunk_id
        
        doc_id = generate_document_id()
        chunk_id = generate_chunk_id(doc_id, 0)
        
        assert len(doc_id) > 0
        assert doc_id in chunk_id
        assert "0" in chunk_id
    
    def test_count_tokens(self):
        """Test token counting"""
        from utils.helpers import count_tokens_approximate
        
        short_text = "Hello world"
        long_text = "This is a much longer text with many words. " * 10
        
        short_count = count_tokens_approximate(short_text)
        long_count = count_tokens_approximate(long_text)
        
        assert long_count > short_count
        assert short_count > 0


# Tests for Models/Schemas
class TestSchemas:
    
    def test_document_chunk_schema(self):
        """Test DocumentChunk schema"""
        from models.schemas import DocumentChunk, ChunkMetadata, SourceType
        from datetime import datetime
        
        metadata = ChunkMetadata(
            document_id="test",
            file_name="test.pdf",
            chunk_id="test_0",
            source_type=SourceType.PDF,
            token_count=500,
            char_count=2000,
            embedding_model="test"
        )
        
        chunk = DocumentChunk(
            chunk_id="test_0",
            document_id="test",
            content="Test content",
            metadata=metadata
        )
        
        assert chunk.chunk_id == "test_0"
        assert chunk.document_id == "test"
    
    def test_rag_response_schema(self):
        """Test RAGResponse schema"""
        from models.schemas import RAGResponse
        
        response = RAGResponse(
            question="Test question?",
            answer="Test answer",
            confidence=0.85,
            processing_time=1.5,
            model="test-model",
            llm_model="llama3.1"
        )
        
        assert response.confidence == 0.85
        assert response.processing_time == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
