"""
Hybrid Semantic Chunking Algorithm
Implements: Document → Structure Detection → Section Segmentation → 
Semantic Boundary Detection → Token-aware Chunk Merge
"""
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import re

from utils.logger import get_logger
from utils.helpers import count_tokens_approximate
from config.settings import (
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, OVERLAP, SIMILARITY_THRESHOLD
)

logger = get_logger(__name__)


@dataclass
class TextSegment:
    """Represents a text segment with metadata"""
    text: str
    segment_type: str  # "heading", "paragraph", "bullet", "table", etc.
    start_index: int
    end_index: int
    token_count: int
    embedding: Optional[np.ndarray] = None


class SemanticChunker:
    """
    Implements hybrid semantic chunking with structure awareness
    and token optimization
    """
    
    def __init__(self, embedding_model=None):
        self.logger = logger
        self.embedding_model = embedding_model
        self.similarity_threshold = SIMILARITY_THRESHOLD
    
    def chunk_document(self, text: str, structured_elements: List[str] = None) -> List[dict]:
        """
        Main chunking pipeline: 
        Document → Structure Detection → Section Segmentation → 
        Semantic Boundary Detection → Token-aware Chunk Merge
        
        Args:
            text: Raw document text
            structured_elements: Pre-extracted structured elements
            
        Returns:
            List of chunks with metadata
        """
        self.logger.info("Starting semantic chunking process...")
        
        # Step 1: Structural Parsing
        segments = self._structural_parsing(text, structured_elements)
        self.logger.info(f"Step 1 - Structural Parsing: Created {len(segments)} segments")
        
        # Step 2: Section Segmentation
        sections = self._section_segmentation(segments)
        self.logger.info(f"Step 2 - Section Segmentation: Created {len(sections)} sections")
        
        # Step 3: Semantic Boundary Detection (if embedding model available)
        if self.embedding_model:
            sections = self._semantic_boundary_detection(sections)
            self.logger.info(f"Step 3 - Semantic Boundary Detection: Refined boundaries")
        
        # Step 4: Token-aware Chunk Merge
        chunks = self._token_aware_merge(sections)
        self.logger.info(f"Step 4 - Token-aware Merge: Created {len(chunks)} final chunks")
        
        # Add chunk IDs and metadata
        final_chunks = self._add_chunk_metadata(chunks)
        
        return final_chunks
    
    def _structural_parsing(self, text: str, structured_elements: List[str] = None) -> List[TextSegment]:
        """
        Step 1: Identify structural elements
        - Headings
        - Paragraphs
        - Bullet Lists
        - Tables
        - Sections
        """
        segments = []
        current_pos = 0
        
        if structured_elements:
            # Use pre-extracted structured elements
            for element in structured_elements:
                segment_type = self._extract_element_type(element)
                text_content = self._extract_element_text(element)
                
                start_idx = text.find(text_content, current_pos)
                if start_idx == -1:
                    start_idx = current_pos
                
                end_idx = start_idx + len(text_content)
                token_count = count_tokens_approximate(text_content)
                
                segments.append(TextSegment(
                    text=text_content,
                    segment_type=segment_type,
                    start_index=start_idx,
                    end_index=end_idx,
                    token_count=token_count
                ))
                
                current_pos = end_idx
        else:
            # Parse text directly
            lines = text.split('\n')
            current_pos = 0
            
            for line in lines:
                if not line.strip():
                    continue
                
                segment_type = self._identify_segment_type(line)
                token_count = count_tokens_approximate(line)
                start_idx = text.find(line, current_pos)
                end_idx = start_idx + len(line) if start_idx != -1 else current_pos + len(line)
                
                segments.append(TextSegment(
                    text=line.strip(),
                    segment_type=segment_type,
                    start_index=start_idx if start_idx != -1 else current_pos,
                    end_index=end_idx,
                    token_count=token_count
                ))
                
                current_pos = end_idx
        
        return segments
    
    def _identify_segment_type(self, text: str) -> str:
        """Identify the type of text segment"""
        text = text.strip()
        
        # Headings
        if text.startswith('#') or (text.isupper() and len(text) < 100):
            return "heading"
        
        # Bullet points
        if re.match(r'^\s*[-•*]\s+', text):
            return "bullet"
        
        # Numbered lists
        if re.match(r'^\s*\d+\.\s+', text):
            return "numbered_list"
        
        # Tables (simplified detection)
        if '|' in text or '\t' in text:
            return "table"
        
        # Default to paragraph
        return "paragraph"
    
    def _extract_element_type(self, element: str) -> str:
        """Extract element type from tagged text"""
        match = re.match(r'\[(\w+)\]', element)
        if match:
            elem_type = match.group(1).lower()
            # Map to our segment types
            type_mapping = {
                'title': 'heading',
                'heading': 'heading',
                'narrative': 'paragraph',
                'list': 'bullet',
                'table': 'table',
            }
            return type_mapping.get(elem_type, 'paragraph')
        return 'paragraph'
    
    def _extract_element_text(self, element: str) -> str:
        """Extract text content from tagged element"""
        return re.sub(r'^\[\w+\]\s*', '', element).strip()
    
    def _section_segmentation(self, segments: List[TextSegment]) -> List[List[TextSegment]]:
        """
        Step 2: Group segments into logical sections
        Sections are typically delimited by headings
        """
        sections = []
        current_section = []
        
        for segment in segments:
            if segment.segment_type == "heading" and current_section:
                # Start new section
                sections.append(current_section)
                current_section = [segment]
            else:
                current_section.append(segment)
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _semantic_boundary_detection(self, sections: List[List[TextSegment]]) -> List[List[TextSegment]]:
        """
        Step 3: Compare adjacent paragraphs using embeddings
        to detect semantic boundaries
        """
        if not self.embedding_model:
            return sections
        
        refined_sections = []
        
        for section in sections:
            if len(section) < 2:
                refined_sections.append(section)
                continue
            
            # Get embeddings for each segment
            embeddings = []
            for segment in section:
                try:
                    embedding = self.embedding_model.embed(segment.text)
                    segment.embedding = embedding
                    embeddings.append(embedding)
                except:
                    embeddings.append(None)
            
            # Detect boundaries based on similarity drops
            current_subsection = [section[0]]
            
            for i in range(1, len(section)):
                if embeddings[i-1] is not None and embeddings[i] is not None:
                    similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
                    
                    # If similarity drops below threshold, start new subsection
                    if similarity < self.similarity_threshold:
                        refined_sections.append(current_subsection)
                        current_subsection = [section[i]]
                    else:
                        current_subsection.append(section[i])
                else:
                    current_subsection.append(section[i])
            
            if current_subsection:
                refined_sections.append(current_subsection)
        
        return refined_sections
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _token_aware_merge(self, sections: List[List[TextSegment]]) -> List[str]:
        """
        Step 4: Merge segments into chunks maintaining:
        - 300-800 tokens per chunk
        - Semantic continuity
        - Overlap for context
        """
        chunks = []
        
        for section in sections:
            current_chunk_segments = []
            current_tokens = 0
            
            for segment in section:
                segment_tokens = segment.token_count
                
                # Check if adding this segment would exceed max
                potential_tokens = current_tokens + segment_tokens + OVERLAP
                
                if potential_tokens <= MAX_CHUNK_SIZE:
                    current_chunk_segments.append(segment)
                    current_tokens += segment_tokens
                else:
                    # Save current chunk if it meets minimum size
                    if current_tokens >= MIN_CHUNK_SIZE:
                        chunk_text = '\n'.join([s.text for s in current_chunk_segments])
                        chunks.append(chunk_text)
                        
                        # Start new chunk with overlap
                        overlap_segments = current_chunk_segments[max(0, len(current_chunk_segments)-2):]
                        current_chunk_segments = overlap_segments + [segment]
                        current_tokens = sum(s.token_count for s in current_chunk_segments)
                    else:
                        # Current chunk too small, just add segment
                        current_chunk_segments.append(segment)
                        current_tokens += segment_tokens
            
            # Don't forget the last chunk
            if current_chunk_segments and current_tokens >= MIN_CHUNK_SIZE:
                chunk_text = '\n'.join([s.text for s in current_chunk_segments])
                chunks.append(chunk_text)
        
        return chunks
    
    def _add_chunk_metadata(self, chunks: List[str]) -> List[dict]:
        """Add metadata to each chunk"""
        final_chunks = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk_dict = {
                "chunk_index": idx,
                "content": chunk_text,
                "token_count": count_tokens_approximate(chunk_text),
                "char_count": len(chunk_text),
                "sentences": len(re.split(r'[.!?]+', chunk_text)),
            }
            final_chunks.append(chunk_dict)
        
        return final_chunks
