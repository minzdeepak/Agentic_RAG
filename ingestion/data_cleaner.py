"""
Data cleaning module for RAG System
"""
import re
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean and preprocess extracted text"""
    
    def __init__(self):
        self.logger = logger
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = self._remove_urls(text)
        
        # Remove email addresses
        text = self._remove_emails(text)
        
        # Remove extra whitespace
        text = self._normalize_whitespace(text)
        
        # Remove special characters (but keep important ones)
        text = self._remove_special_chars(text)
        
        # Remove duplicate lines
        text = self._remove_duplicate_lines(text)
        
        # Normalize line breaks
        text = self._normalize_line_breaks(text)
        
        self.logger.info(f"Text cleaned successfully. Original length: {len(text)}")
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple spaces with single space
        text = re.sub(r'  +', ' ', text)
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove problematic special characters while preserving important ones"""
        # Keep alphanumeric, common punctuation, and whitespace
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        return text
    
    def _remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines"""
        lines = text.split('\n')
        unique_lines = []
        prev_line = None
        
        for line in lines:
            if line.strip() != prev_line:
                unique_lines.append(line)
                prev_line = line.strip()
        
        return '\n'.join(unique_lines)
    
    def _normalize_line_breaks(self, text: str) -> str:
        """Normalize line breaks to \n"""
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text
    
    def clean_chunks(self, chunks: List[str]) -> List[str]:
        """
        Clean multiple text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of cleaned chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                cleaned = self.clean_text(chunk)
                if cleaned.strip():  # Skip if becomes empty after cleaning
                    cleaned_chunks.append(cleaned)
        
        self.logger.info(f"Cleaned {len(cleaned_chunks)} chunks from {len(chunks)}")
        return cleaned_chunks
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text (headers, footers, etc.)
        """
        # Remove common page headers/footers
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip common boilerplate
            if any(skip in line_lower for skip in [
                'page', 'of page', 'confidential', 'do not distribute',
                'copyright', '©', '®', 'all rights reserved',
                'phone:', 'fax:', 'visit us at'
            ]):
                continue
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def extract_structure(self, text: str) -> Tuple[List[str], dict]:
        """
        Extract document structure (headings, sections, etc.)
        
        Args:
            text: Raw text with potential structure markers
            
        Returns:
            Tuple of (sections, structure_metadata)
        """
        sections = []
        structure_metadata = {
            "headings": [],
            "has_numbering": False,
            "has_bullets": False,
            "list_depth": 0
        }
        
        lines = text.split('\n')
        current_section = []
        
        for line in lines:
            # Detect headings (capitalized lines, markdown, or numbered)
            if self._is_heading(line):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                structure_metadata["headings"].append(line)
                current_section.append(line)
            
            # Detect bullet points
            if self._is_bullet(line):
                structure_metadata["has_bullets"] = True
            
            # Detect numbered lists
            if self._is_numbered(line):
                structure_metadata["has_numbering"] = True
            
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections, structure_metadata
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is a heading"""
        line = line.strip()
        
        # Markdown headings
        if line.startswith('#'):
            return True
        
        # All caps with reasonable length
        if line.isupper() and len(line) > 3 and len(line) < 100:
            return True
        
        # Numbered headings
        if re.match(r'^\d+\.\s+[A-Z]', line):
            return True
        
        return False
    
    def _is_bullet(self, line: str) -> bool:
        """Check if line is a bullet point"""
        return bool(re.match(r'^\s*[-•*]\s+', line.strip()))
    
    def _is_numbered(self, line: str) -> bool:
        """Check if line is a numbered item"""
        return bool(re.match(r'^\s*\d+\.\s+', line.strip()))
