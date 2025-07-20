import logging
import asyncio
from typing import List, Dict, Any, Literal
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .embedding_manager import EmbeddingManager
from .document_processor import DocumentProcessor
from .qdrant_manager import QdrantManager

class DocumentIndexer:
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "documents",
                 distance_threshold: float = 0.7,  # Restored to proper value
                 embedding_provider: Literal["fastembed", "deepinfra"] = "fastembed",
                 force_recreate: bool = False):
        
        logging.info("Initializing DocumentIndexer...")
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(embedding_provider)
        self.document_processor = DocumentProcessor()
        self.qdrant_manager = QdrantManager(qdrant_host, qdrant_port, collection_name)
        
        self.collection_name = collection_name
        self.distance_threshold = distance_threshold
        
        # Store initialization parameters for async initialization
        self._force_recreate = force_recreate
        self._initialized = False
        self.vector_store = None

        logging.info("DocumentIndexer initialized (async initialization pending)")

    async def _initialize_vector_store(self, force_recreate: bool = False):
        """Async initialization of vector store."""
        if self._initialized and not force_recreate:
            return
            
        vector_size = self.embedding_manager.get_embedding_dimension()
        
        if not await self.qdrant_manager.collection_exists() or force_recreate:
            await self.qdrant_manager.create_collection(vector_size, force_recreate)
        
        # Create sync client for QdrantVectorStore using the same connection details
        try:
            from qdrant_client import QdrantClient
            
            # Try gRPC first, then fallback to HTTP (same as async client)
            try:
                sync_client = QdrantClient(host=self.qdrant_manager.host, grpc_port=6334, prefer_grpc=True)
                logging.info("Using gRPC connection for sync client")
            except Exception:
                sync_client = QdrantClient(host=self.qdrant_manager.host, port=6333)
                logging.info("Using HTTP connection for sync client")
            
            self.vector_store = QdrantVectorStore(
                client=sync_client,
                collection_name=self.collection_name,
                embedding=self.embedding_manager.embeddings,
            )
            
            self._initialized = True
            logging.info(f"Vector store initialized successfully for collection '{self.collection_name}'")
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            self._initialized = False
            raise
    
    async def index_documents(self, file_paths: List[str], user_id:str, thread_id: str) -> Dict[str, Any]:
        # Ensure vector store is initialized
        if not self._initialized:
            await self._initialize_vector_store(self._force_recreate)
            
        logging.info(f"Indexing {len(file_paths)} documents for user '{user_id}' in thread '{thread_id}'")
        
        # Filter out files that already exist
        files_to_process = []
        skipped_files = []
        
        # Check file existence in parallel
        existence_tasks = [self.qdrant_manager.document_exists(
            self.document_processor.calculate_file_hash(file_path),
            user_id,
            thread_id
        ) for file_path in file_paths]
        
        # Also check global existence for debugging
        global_existence_tasks = [self.qdrant_manager.document_exists_globally(
            self.document_processor.calculate_file_hash(file_path)
        ) for file_path in file_paths]
        
        existence_results = await asyncio.gather(*existence_tasks, return_exceptions=True)
        global_existence_results = await asyncio.gather(*global_existence_tasks, return_exceptions=True)
        
        for i, (file_path, exists, global_exists) in enumerate(zip(file_paths, existence_results, global_existence_results)):
            if isinstance(exists, Exception):
                logging.error(f"Error checking file existence for {file_path}: {exists}")
                files_to_process.append(file_path)
            elif exists:
                logging.info(f"Skipping existing file '{file_path}' in thread '{thread_id}' (already indexed)")
                skipped_files.append(file_path)
            else:
                if isinstance(global_exists, bool) and global_exists:
                    logging.info(f"Processing file '{file_path}' for thread '{thread_id}' (duplicating from another thread)")
                else:
                    logging.info(f"Processing new file '{file_path}' for thread '{thread_id}' (first time indexing)")
                files_to_process.append(file_path)
        
        if not files_to_process:
            return {
                "message": f"All {len(file_paths)} files already indexed in thread '{thread_id}'",
                "indexed_count": 0,
                "skipped_count": len(skipped_files),
                "user_id": user_id,
                "thread_id": thread_id,
            }
        
        # Process documents in parallel
        async def process_file(file_path: str) -> List[Document]:
            try:
                documents = self.document_processor.load_document(file_path, user_id, thread_id)
                logging.info(f"Loaded {len(documents)} chunks from {file_path}")
                return documents
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                return []
        
        processing_tasks = [process_file(file_path) for file_path in files_to_process]
        all_document_lists = await asyncio.gather(*processing_tasks)
        
        all_documents = []
        for doc_list in all_document_lists:
            all_documents.extend(doc_list)
        
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
            if self.vector_store is not None:
                self.vector_store.add_documents(all_documents)
                indexed_count = len(all_documents)
                logging.info(f"Upload completed: {indexed_count} chunks")
            else:
                logging.error("Vector store is not initialized")
                indexed_count = 0
            
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
        
    async def search(self, query: str, user_id: str, thread_id: str, top_k: int = 10) -> List[Document]:
        # Ensure vector store is initialized
        if not self._initialized:
            try:
                await self._initialize_vector_store(self._force_recreate)
            except Exception as e:
                logging.error(f"Failed to initialize vector store for search: {e}")
                return []
            
        logging.info(f"Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

        try:
            if self.vector_store is None:
                logging.error("Vector store is not initialized")
                return []
                
            search_k = min(top_k * 2, 30)
            results = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=search_k
            )

            filtered_results = []
            for doc, score in results:
                if score >= self.distance_threshold:  # e.g., 0.7
                    doc_user_id = doc.metadata.get('user_id')
                    doc_thread_id = doc.metadata.get('thread_id')

                    if doc_user_id == user_id and doc_thread_id == thread_id:
                        filtered_results.append(doc)
                        if len(filtered_results) >= top_k:
                            break
                    else:
                        logging.debug(f"Metadata mismatch - Expected: user={user_id}, thread={thread_id}, "
                                    f"Got: user={doc_user_id}, thread={doc_thread_id}")
                else:
                    logging.debug(f"Filtered out - Relevance Score: {score:.3f} < threshold: {self.distance_threshold}")

            logging.info(f"Search completed: {len(filtered_results)}/{len(results)} results passed filtering")
            return filtered_results

        except Exception as e:
            logging.error(f"Search failed: {e}")
            # Return empty list instead of crashing
            return []
    
    # Sync wrappers for backward compatibility
    def index_documents_sync(self, file_paths: List[str], user_id:str, thread_id: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.index_documents(file_paths, user_id, thread_id))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.index_documents(file_paths, user_id, thread_id))
    
    def search_sync(self, query: str, user_id: str, thread_id: str, top_k: int = 10) -> List[Document]:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.search(query, user_id, thread_id, top_k))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.search(query, user_id, thread_id, top_k))
    

