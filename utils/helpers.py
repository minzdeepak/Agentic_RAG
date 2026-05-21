"""
Helper utilities for RAG System
"""
import uuid
from datetime import datetime
import hashlib
from typing import List
import json


def generate_document_id() -> str:
    """Generate unique document ID"""
    return str(uuid.uuid4())


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate unique chunk ID"""
    return f"{document_id}_{chunk_index}"


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count using simple word splitting
    More accurate counting would require tiktoken or similar
    """
    words = text.split()
    # Rough estimate: ~1.3 tokens per word for English
    return len(words) * 4 // 3


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file"""
    return hashlib.sha256(file_bytes).hexdigest()


def format_response_time(seconds: float) -> str:
    """Format response time for display"""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    return f"{seconds:.2f}s"


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return text.strip()


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def serialize_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.utcnow().isoformat()


def parse_metadata(metadata_str: str) -> dict:
    """Parse metadata string to dict"""
    try:
        return json.loads(metadata_str) if metadata_str else {}
    except:
        return {}
