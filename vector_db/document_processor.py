import hashlib
import logging
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

class DocumentProcessor:
    """Simple document processor for PDF files."""
    
    def __init__(self):
        logging.info("Initialized document processor")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_document(self, file_path: str, user_id: str, thread_id: str) -> List[Document]:
        """Load and process a single PDF document."""
        logging.info(f"Loading document: {file_path}")
        
        file_path_obj = Path(file_path)
        
        # Only support PDF files for now
        if file_path_obj.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported. Got: {file_path_obj.suffix}")
        
        # Calculate file hash for deduplication
        file_hash = self.calculate_file_hash(file_path)
        
        try:
            # Use UnstructuredPDFLoader with specified settings
            loader = UnstructuredPDFLoader(
                str(file_path),
                strategy="fast",
                chunking_strategy="by_title",
                mode="elements",
                max_characters=3000,
                overlap=100,
                combine_text_under_n_chars=500
            )
            
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} chunks from PDF")
            
            # Add metadata to each chunk
            for i, doc in enumerate(documents):
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                doc.metadata.update({
                    'user_id': user_id,
                    'thread_id': thread_id,
                    'file_hash': file_hash,
                    'content_hash': content_hash,
                    'source_file': str(file_path_obj),
                    'file_type': '.pdf',
                    'chunk_index': i,
                    'chunk_size': len(doc.page_content)
                })
            
            logging.info(f"Successfully processed {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return []
