"""
Example usage scripts for RAG System
"""

# ============================================
# Example 1: Basic Document Ingestion
# ============================================

def example_ingest_pdf():
    """Ingest a PDF document"""
    from ingestion.ingestion_pipeline import IngestionPipeline
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chromadb_store import ChromaVectorStore
    from models.schemas import SourceType
    
    # Initialize components
    embedding_model = EmbeddingModel()
    vector_store = ChromaVectorStore()
    pipeline = IngestionPipeline(embedding_model, vector_store)
    
    # Ingest document
    result = pipeline.ingest_document(
        file_path="path/to/document.pdf",
        source_type=SourceType.PDF,
        document_metadata={
            "category": "finance",
            "year": 2024
        }
    )
    
    print(f"✅ Document ingested successfully!")
    print(f"   Document ID: {result['document_id']}")
    print(f"   Chunks created: {result['chunks_created']}")
    print(f"   Total tokens: {result['total_tokens']}")


# ============================================
# Example 2: Query the RAG System
# ============================================

def example_query():
    """Query the RAG system for answers"""
    from ingestion.ingestion_pipeline import IngestionPipeline
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chromadb_store import ChromaVectorStore
    from retrieval.hybrid_retriever import HybridRetriever
    from llm.ollama_llm import OllamaLLM
    
    # Initialize components
    embedding_model = EmbeddingModel()
    vector_store = ChromaVectorStore()
    retriever = HybridRetriever(embedding_model, vector_store)
    llm = OllamaLLM()
    
    # Query
    question = "What are the main findings from the document?"
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(question, top_k=3, use_hybrid=True)
    
    print(f"📚 Retrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n   Document {i}:")
        print(f"   Score: {doc.get('combined_score', 0):.3f}")
        print(f"   Preview: {doc['content'][:100]}...")
    
    # Generate answer
    context = [doc['content'] for doc in retrieved_docs]
    answer_response = llm.generate_answer(
        query=question,
        context=context
    )
    
    print(f"\n💡 Generated Answer:")
    print(answer_response['answer'])


# ============================================
# Example 3: Batch Document Ingestion
# ============================================

def example_batch_ingest():
    """Ingest multiple documents at once"""
    from ingestion.ingestion_pipeline import IngestionPipeline
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chromadb_store import ChromaVectorStore
    from models.schemas import SourceType
    from pathlib import Path
    
    # Initialize pipeline
    embedding_model = EmbeddingModel()
    vector_store = ChromaVectorStore()
    pipeline = IngestionPipeline(embedding_model, vector_store)
    
    # Prepare file list
    documents = [
        ("document1.pdf", SourceType.PDF),
        ("document2.docx", SourceType.DOCX),
        ("document3.txt", SourceType.TXT),
    ]
    
    # Batch ingest
    results = pipeline.ingest_batch(documents)
    
    print("📁 Batch Ingestion Results:")
    for result in results:
        if result['status'] == 'success':
            print(f"✅ {result['file_name']}: {result['chunks_created']} chunks")
        else:
            print(f"❌ {result['file_name']}: {result['message']}")


# ============================================
# Example 4: Advanced Query with Metadata Filtering
# ============================================

def example_advanced_query():
    """Query with metadata filtering"""
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chromadb_store import ChromaVectorStore
    from retrieval.hybrid_retriever import HybridRetriever
    
    embedding_model = EmbeddingModel()
    vector_store = ChromaVectorStore()
    retriever = HybridRetriever(embedding_model, vector_store)
    
    # Query
    question = "Financial performance in 2024"
    
    # Generate query embedding
    query_embedding = embedding_model.embed_query(question)
    
    # Search with metadata filtering
    results = vector_store.search(
        query_embedding,
        n_results=5,
        where={"source_type": "pdf"}  # Only PDF documents
    )
    
    print(f"🔍 Search Results (PDF only):")
    print(f"   Found {len(results['ids'][0])} documents")


# ============================================
# Example 5: Custom Chunking Strategy
# ============================================

def example_custom_chunking():
    """Test custom chunking strategy"""
    from chunking.semantic_chunker import SemanticChunker
    from embeddings.embedding_model import EmbeddingModel
    
    embedding_model = EmbeddingModel()
    chunker = SemanticChunker(embedding_model)
    
    # Sample text
    text = """
    Chapter 1: Introduction
    
    This document covers important topics. Here are some key points:
    - Point 1
    - Point 2
    - Point 3
    
    Chapter 2: Main Content
    
    The main content discusses...
    """
    
    # Chunk text
    chunks = chunker.chunk_document(text)
    
    print(f"🔀 Chunking Results:")
    print(f"   Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   Chunk {i}:")
        print(f"   Tokens: {chunk['token_count']}")
        print(f"   Preview: {chunk['content'][:50]}...")


# ============================================
# Example 6: System Health Check
# ============================================

def example_health_check():
    """Check system health and components"""
    from embeddings.embedding_model import EmbeddingModel
    from vectordb.chromadb_store import ChromaVectorStore
    from retrieval.hybrid_retriever import HybridRetriever
    from llm.ollama_llm import OllamaLLM
    
    print("🏥 System Health Check\n")
    
    try:
        # Check embeddings
        embedding_model = EmbeddingModel()
        print(f"✅ Embeddings: {embedding_model.get_model_info()['model_name']}")
    except Exception as e:
        print(f"❌ Embeddings: {str(e)}")
    
    try:
        # Check vector DB
        vector_store = ChromaVectorStore()
        info = vector_store.get_collection_info()
        print(f"✅ Vector DB: {info['document_count']} documents")
    except Exception as e:
        print(f"❌ Vector DB: {str(e)}")
    
    try:
        # Check LLM
        llm = OllamaLLM()
        models = llm.list_available_models()
        print(f"✅ LLM: {llm.model} ({len(models)} models available)")
    except Exception as e:
        print(f"❌ LLM: {str(e)}")


# ============================================
# Example 7: Direct API Usage
# ============================================

def example_api_usage():
    """Use the system via FastAPI"""
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    # Health check
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")
    
    # Query
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        json={
            "question": "What is the main topic?",
            "top_k": 3,
            "include_sources": True,
            "use_hybrid_search": True
        }
    )
    result = response.json()
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Ingest PDF", example_ingest_pdf),
        "2": ("Query RAG", example_query),
        "3": ("Batch Ingest", example_batch_ingest),
        "4": ("Advanced Query", example_advanced_query),
        "5": ("Custom Chunking", example_custom_chunking),
        "6": ("Health Check", example_health_check),
        "7": ("API Usage", example_api_usage),
    }
    
    print("🚀 RAG System Examples\n")
    print("Choose an example:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n▶️  Running: {name}\n")
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
