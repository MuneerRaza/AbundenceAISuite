import hashlib
import logging
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, # Changed back for PDFs
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredODTLoader,
    TextLoader,
    JSONLoader,
    CSVLoader
)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.ppt', '.pptx', 
        '.csv', '.xlsx', '.xls', '.odt', '.txt',
        '.json', '.config', '.yaml', '.yml',
        '.py', '.js', '.ts', '.html', '.css', '.java',
        '.cpp', '.c', '.h', '.cs', '.php', '.rb',
        '.go', '.rs', '.swift', '.kt', '.scala',
        '.r', '.sql', '.sh', '.bat', '.ps1',
        '.xml', '.md', '.rst', '.toml', '.ini'
    }
    
    def calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_appropriate_loader(self, file_path_str: str, file_extension: str):
        if file_extension == '.pdf':
            return UnstructuredPDFLoader(file_path_str, mode="single")
        elif file_extension in ['.docx', '.doc']:
            return UnstructuredWordDocumentLoader(file_path_str, mode="single")
        elif file_extension in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path_str, mode="single")
        elif file_extension == '.odt':
            return UnstructuredODTLoader(file_path_str, mode="single")
        elif file_extension in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path_str, mode="single")
        elif file_extension == '.csv':
            return CSVLoader(file_path_str, encoding='utf-8', autodetect_encoding=True)
        elif file_extension == '.json':
            return JSONLoader(file_path_str, jq_schema='.', text_content=False)
        else: # Covers all other text-based formats
            return TextLoader(file_path_str, encoding='utf-8', autodetect_encoding=True)

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def load_document(self, file_path: str, user_id: str, thread_id: str) -> List[Document]:
        file_path_obj = Path(file_path)
        if not file_path_obj.is_file():
            logging.error(f"File not found or is not a file: {file_path}")
            return []

        file_extension = file_path_obj.suffix.lower()
        if not self.is_supported_file(str(file_path_obj)):
            logging.warning(f"Unsupported file type: {file_extension}. Skipping.")
            return []
        
        logging.info(f"Processing '{file_path_obj.name}' with hybrid strategy.")
        
        try:
            loader = self._get_appropriate_loader(str(file_path_obj), file_extension)
            docs_from_loader = loader.load()
            final_docs = []

            logging.info(f"Non-PDF detected. Using whole-document splitting.")
            full_text = "\n\n".join(doc.page_content for doc in docs_from_loader if doc.page_content)
            if full_text:
                text_splitter = self._get_text_splitter()
                final_docs = text_splitter.create_documents([full_text])

            if not final_docs:
                logging.warning(f"No content processed for {file_path_obj.name}. Skipping.")
                return []

            file_hash = self.calculate_file_hash(str(file_path_obj))
            total_chunks = len(final_docs)
            for i, doc in enumerate(final_docs):
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                new_metadata = doc.metadata.copy()
                new_metadata.update({
                    'user_id': user_id,
                    'thread_id': thread_id,
                    'file_hash': file_hash,
                    'content_hash': content_hash,
                    'source_file': str(file_path_obj),
                    'chunk_number': i + 1,
                    'document_total_chunks': total_chunks
                })
                doc.metadata = new_metadata

            logging.info(f"Successfully processed '{file_path_obj.name}', created {total_chunks} chunks.")
            return final_docs

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}", exc_info=True)
            return []
    
    def is_supported_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_supported_extensions(self) -> List[str]:
        return sorted(list(self.SUPPORTED_EXTENSIONS))