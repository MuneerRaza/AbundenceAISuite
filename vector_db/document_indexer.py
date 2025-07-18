import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore

from .embedding_manager import EmbeddingManager
from .document_processor import DocumentProcessor
from .qdrant_manager import QdrantManager

class DocumentIndexer:
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "documents",
                 similarity_threshold: float = 0.3,
                 force_recreate: bool = False):
        
        logging.info("Initializing DocumentIndexer...")
        
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.document_processor = DocumentProcessor()
        self.qdrant_manager = QdrantManager(qdrant_host, qdrant_port, collection_name)
        
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Initialize vector store
        self._initialize_vector_store(force_recreate)

        logging.info("DocumentIndexer initialized")

    def _initialize_vector_store(self, force_recreate: bool = False):
        vector_size = self.embedding_manager.get_embedding_dimension()
        
        if not self.qdrant_manager.collection_exists() or force_recreate:
            self.qdrant_manager.create_collection(vector_size, force_recreate)
        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_manager.client,
            collection_name=self.collection_name,
            embedding=self.embedding_manager.embeddings,
        )
    
    def index_documents(self, file_paths: List[str], user_id:str, thread_id: str) -> Dict[str, Any]:
        logging.info(f"Indexing {len(file_paths)} documents for user '{user_id}' in thread '{thread_id}'")
        
        # Filter out files that already exist
        files_to_process = []
        skipped_files = []
        
        for file_path in file_paths:
            file_hash = self.document_processor.calculate_file_hash(file_path)
            if self.qdrant_manager.document_exists(file_hash):
                logging.info(f"Skipping existing file: {file_path}")
                skipped_files.append(file_path)
            else:
                files_to_process.append(file_path)
        
        if not files_to_process:
            return {
                "message": f"All {len(file_paths)} files already indexed",
                "indexed_count": 0,
                "skipped_count": len(skipped_files),
                "user_id": user_id,
                "thread_id": thread_id,
            }
        
        all_documents = []
        for file_path in files_to_process:
            try:
                documents = self.document_processor.load_document(file_path, user_id, thread_id)
                all_documents.extend(documents)
                logging.info(f"Loaded {len(documents)} chunks from {file_path}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        if not all_documents:
            return {
                "message": "No documents were loaded",
                "indexed_count": 0,
                "skipped_count": len(skipped_files),
                "user_id": user_id,
                "thread_id": thread_id
            }
        
        try:
            logging.info(f"Uploading {len(all_documents)} document chunks to Qdrant...")
            self.vector_store.add_documents(all_documents)
            indexed_count = len(all_documents)
            logging.info(f"Upload completed: {indexed_count} chunks")
            
        except Exception as e:
            logging.error(f"Upload failed: {e}")
            indexed_count = 0
        
        logging.info(f"Indexing completed: {indexed_count} chunks indexed, "
                     f"{len(skipped_files)} files skipped")

        return {
            "message": f"Successfully indexed {indexed_count} document chunks from {len(files_to_process)} files",
            "indexed_count": indexed_count,
            "skipped_count": len(skipped_files),
            "user_id": user_id,
            "thread_id": thread_id
        }

    def search(self, query: str, user_id: str, thread_id: str, top_k: int = 10) -> List[Document]:
        """Search for similar documents using vector similarity."""
        logging.info(f"Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")
        
        try:
            # Perform vector search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k * 2  # Get more results for filtering
            )
            
            # Filter by similarity threshold and optionally by thread_id
            filtered_results = []
            for doc, score in results:
                if score >= self.similarity_threshold:
                    if user_id == doc.metadata.get('user_id') and thread_id == doc.metadata.get('thread_id'):
                        filtered_results.append(doc)
                        if len(filtered_results) >= top_k:
                            break
                    else:
                        logging.debug(f"Unmatched metadata for doc: {doc.metadata}")

            logging.info(f"Search completed: {len(filtered_results)}/{len(results)} results")
            return filtered_results
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

    # def delete_all_collections(self):
    #     """Delete all collections in Qdrant."""
    #     start_time = time.time()
    #     try:
    #         collections = self.qdrant_manager.client.get_collections().collections
    #         deleted_count = 0
            
    #         for collection in collections:
    #             try:
    #                 self.qdrant_manager.client.delete_collection(collection.name)
    #                 deleted_count += 1
    #                 logging.info(f"Deleted collection: {collection.name}")
    #             except Exception as e:
    #                 logging.error(f"Error deleting collection {collection.name}: {e}")
            
    #         # Reinitialize vector store after deleting all collections
    #         self._initialize_vector_store()
            
    #         total_time = time.time() - start_time
    #         logging.info(f"All collections deleted: {deleted_count} collections ({total_time:.3f}s)")
            
    #     except Exception as e:
    #         elapsed = time.time() - start_time
    #         logging.error(f"Error deleting all collections ({elapsed:.3f}s): {e}")
    

