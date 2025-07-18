"""
Example usage of the simplified vector database components.
"""

import logging
import sys
import os
from vector_db import get_global_indexer

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    """Example usage of the document indexer."""
    setup_logging()
    
    # Set cache directory for FastEmbed
    os.environ['FASTEMBED_CACHE_PATH'] = 'cache'
    
    # Initialize the document indexer
    indexer = get_global_indexer()

    
    logging.info("Document indexer initialized successfully")
    
    # Example: Index a PDF document
    pdf_files = [
        r"C:\Users\MuneerRaza\Downloads\CognifootAI_FYP_report_final.pdf"
    ]
    
    if pdf_files:
        user_id = "test_user"
        thread_id = "b"
        
        # Index documents
        logging.info("Starting document indexing...")
        result = indexer.index_documents(pdf_files, user_id, thread_id)
        logging.info(f"Indexing result: {result}")
        
        # Search for documents
        query = "Supervisor"
        logging.info(f"Searching for: {query}")
        search_results = indexer.search(query, user_id, thread_id, top_k=5)
        
        logging.info(f"Found {len(search_results)} results:")
        for i, doc in enumerate(search_results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Chunk: {doc.metadata.get('chunk_index', 'Unknown')}")

    
    else:
        logging.info("No PDF files specified. Add file paths to pdf_files list to test.")


if __name__ == "__main__":
    
    # os.environ['FASTEMBED_CACHE_PATH'] = 'cache'
    # indexer = get_global_indexer()
    # indexer.qdrant_manager.delete_collection() 
    
    main()
