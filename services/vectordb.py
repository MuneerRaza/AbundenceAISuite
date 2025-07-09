import os
import hashlib
import time
import logging
from typing import List, Literal, Tuple, Optional
from pathlib import Path
import concurrent.futures

# Qdrant specific imports
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode

# LangChain Caching, Storage, and Retrievers
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import CacheBackedEmbeddings

# LangChain Embeddings and Document Loaders
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_deepinfra import DeepInfraEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader,
    UnstructuredODTLoader, UnstructuredExcelLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, TextLoader, JSONLoader
)


class OptimizedDocumentIndexer:
    """
    A high-performance, high-quality document indexing and retrieval system.
    Features:
    - Multiple retrieval strategies (basic, hierarchical)
    - Configurable chunking strategies
    - Caching for embeddings to avoid re-computation
    - Parallelized document processing for rapid indexing
    - Local persistent Qdrant storage with detailed timing logs
    """
    LOADER_MAPPING = {
        ".pdf": (PyPDFLoader, {}), ".pptx": (UnstructuredPowerPointLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}), ".docx": (UnstructuredWordDocumentLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}), ".odt": (UnstructuredODTLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}), ".xls": (UnstructuredExcelLoader, {}),
        ".csv": (UnstructuredExcelLoader, {}), ".html": (UnstructuredHTMLLoader, {}),
        ".htm": (UnstructuredHTMLLoader, {}), ".md": (UnstructuredMarkdownLoader, {}),
        ".json": (JSONLoader, {}), ".py": (TextLoader, {"encoding": "utf-8"}),
        ".js": (TextLoader, {"encoding": "utf-8"}), ".java": (TextLoader, {"encoding": "utf-8"}),
        ".cs": (TextLoader, {"encoding": "utf-8"}), ".cpp": (TextLoader, {"encoding": "utf-8"}),
        ".c": (TextLoader, {"encoding": "utf-8"}), ".h": (TextLoader, {"encoding": "utf-8"}),
        ".go": (TextLoader, {"encoding": "utf-8"}), ".rb": (TextLoader, {"encoding": "utf-8"}),
        ".php": (TextLoader, {"encoding": "utf-8"}), ".swift": (TextLoader, {"encoding": "utf-8"}),
        ".kt": (TextLoader, {"encoding": "utf-8"}), ".ts": (TextLoader, {"encoding": "utf-8"}),
        ".sql": (TextLoader, {"encoding": "utf-8"}), ".sh": (TextLoader, {"encoding": "utf-8"}),
        ".yml": (TextLoader, {"encoding": "utf-8"}), ".yaml": (TextLoader, {"encoding": "utf-8"}),
        ".txt": (TextLoader, {"encoding": "utf-8"}),
    }

    def __init__(
        self,
        collection_name: str,
        embedding_provider: Literal["fastembed", "deepinfra"] = "fastembed",
        qdrant_path: str = "./qdrant",
        cache_path: str = "./cache",
        strategy: Literal["basic", "hierarchical"] = "basic",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.cache_path = cache_path
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize Qdrant client with local persistent storage
        try:
            # Create Qdrant data directory if it doesn't exist
            os.makedirs(self.qdrant_path, exist_ok=True)
            
            start_time = time.time()
            self.client = QdrantClient(path=self.qdrant_path)
            connection_time = time.time() - start_time
            
            # Test connection by getting collections
            self.client.get_collections()
            self.logger.info(f"✅ Connected to local Qdrant storage at '{self.qdrant_path}' (connection time: {connection_time:.3f}s)")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to local Qdrant storage at '{self.qdrant_path}': {e}")
            
        # Create cache directory
        os.makedirs(self.cache_path, exist_ok=True)

        # Initialize embeddings with caching
        try:
            start_time = time.time()
            fs_cache = LocalFileStore(os.path.join(self.cache_path, "embeddings"))
            core_embed_model = self._get_embedding_model(embedding_provider)
            
            # Create a namespace for the cache based on the embedding provider and model
            namespace = f"{embedding_provider}_{getattr(core_embed_model, 'model_name', 'default')}"
            
            self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                core_embed_model, fs_cache, namespace=namespace
            )
            embedding_init_time = time.time() - start_time
            self.logger.info(f"✅ Initialized embeddings with caching (initialization time: {embedding_init_time:.3f}s)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")

        # Initialize vector store with vector retrieval mode (hybrid requires sparse embeddings)
        # We'll create it lazily when the first document is added
        self.vector_store = None
        self._vector_store_config = {
            "client": self.client,
            "collection_name": self.collection_name,
            "embedding": self.cached_embedder,
            "retrieval_mode": RetrievalMode.DENSE,  # Use dense vector search (can be combined with filters)
        }
        
        self.logger.info(f"✅ Vector store configuration prepared (will be created when documents are added)")

        # Initialize retriever based on strategy
        try:
            start_time = time.time()
            self._initialize_retriever()
            retriever_init_time = time.time() - start_time
            self.logger.info(f"✅ Initialized retriever (initialization time: {retriever_init_time:.3f}s)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {e}")

    def _collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _create_collection(self):
        """Create a new collection in Qdrant with appropriate settings."""
        try:
            # Get vector size from the embedding model
            sample_embedding = self.cached_embedder.embed_query("test")
            vector_size = len(sample_embedding)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")
        except Exception as e:
            raise RuntimeError(f"Failed to create collection '{self.collection_name}': {e}")

    def _ensure_vector_store(self):
        """Ensure vector store is initialized (lazy initialization)."""
        if self.vector_store is None:
            try:
                start_time = time.time()
                
                # Ensure collection exists before creating vector store
                if not self._collection_exists():
                    self.logger.info(f"Creating collection '{self.collection_name}'...")
                    self._create_collection()
                
                self.vector_store = QdrantVectorStore(**self._vector_store_config)
                init_time = time.time() - start_time
                self.logger.info(f"✅ Created vector store (initialization time: {init_time:.3f}s)")
            except Exception as e:
                raise RuntimeError(f"Failed to create vector store: {e}")
        return self.vector_store

    def _ensure_retriever(self):
        """Ensure retriever is initialized (lazy initialization)."""
        if not hasattr(self, 'retriever'):
            vector_store = self._ensure_vector_store()
            if self.strategy == "basic":
                self.retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            elif self.strategy == "hierarchical":
                self.retriever = ParentDocumentRetriever(
                    vectorstore=vector_store,
                    docstore=self.doc_store,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter
                )
        return self.retriever

    def _initialize_retriever(self):
        """Initialize the appropriate retriever based on the strategy."""
        if self.strategy == "basic":
            # Basic retrieval: Direct vector search with optimal chunk sizes
            # We'll create the retriever lazily when needed
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
        elif self.strategy == "hierarchical":
            # Hierarchical: Small chunks for search, medium chunks for context
            self.doc_store = InMemoryStore()
            # We'll create the ParentDocumentRetriever lazily when needed
            self.child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,  # Small chunks for precise search
                chunk_overlap=50
            )
            self.parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,  # Medium-sized parents
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        self.logger.info(f"✅ Initialized {self.strategy} retrieval strategy with dense vector search")

    def _get_embedding_model(self, provider: str):
        """Get the appropriate embedding model based on provider."""
        if provider == "deepinfra":
            api_key = os.getenv("DEEPINFRA_API_TOKEN")
            if not api_key: 
                raise ValueError("DEEPINFRA_API_TOKEN environment variable not set.")
            try:
                return DeepInfraEmbeddings(model="BAAI/bge-m3")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DeepInfra embeddings: {e}")
                self.logger.info("Falling back to FastEmbed embeddings...")
                return FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en")
        else:
            try:
                return FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en")
            except Exception as e:
                self.logger.error(f"Failed to initialize FastEmbed embeddings: {e}")
                raise RuntimeError(f"Could not initialize any embedding model: {e}")

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (IOError, OSError) as e:
            logging.error(f"Error reading file {file_path}: {e}")
            raise

    def _load_and_prep_document(self, file_path: Path, thread_id: str) -> List[Document]:
        """Load and prepare a document based on the strategy."""
        start_time = time.time()
        
        ext = file_path.suffix.lower()
        if ext not in self.LOADER_MAPPING: 
            return []
        
        try:
            loader_class, loader_args = self.LOADER_MAPPING[ext]
            documents = loader_class(str(file_path), **loader_args).load()
            
            load_time = time.time() - start_time
            self.logger.info(f"📄 Document loading time for '{file_path.name}': {load_time:.3f}s")
            
            file_hash = self._calculate_file_hash(str(file_path))
            
            processed_docs = []
            for doc in documents:
                doc.metadata["file_hash"] = file_hash
                doc.metadata["thread_id"] = thread_id
                doc.metadata["source_file"] = str(file_path)
                
                # For basic strategy, split documents into chunks immediately
                if self.strategy == "basic":
                    chunks = self.text_splitter.split_documents([doc])
                    for i, chunk in enumerate(chunks):
                        chunk.metadata["chunk_index"] = i
                        chunk.metadata["total_chunks"] = len(chunks)
                    processed_docs.extend(chunks)
                else:
                    # For hierarchical, let ParentDocumentRetriever handle splitting
                    processed_docs.append(doc)
                    
            return processed_docs
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            return []

    def _filter_new_files(self, file_paths: List[Path]) -> List[Path]:
        """Filters a list of files, returning only those not already indexed."""
        existing_hashes = self._get_all_hashes_in_collection()
        new_files = []
        
        # Calculate hashes in parallel for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_path = {
                executor.submit(self._calculate_file_hash, str(p)): p 
                for p in file_paths
            }
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    file_hash = future.result()
                    if file_hash not in existing_hashes:
                        new_files.append(path)
                    else:
                        self.logger.info(f"Skipping cached document: {path.name}")
                except Exception as exc:
                    self.logger.error(f"Error calculating hash for {path.name}: {exc}")
        return new_files
    
    def index_files(self, file_paths: List[str], thread_id: str):
        """
        Processes and indexes a list of files in parallel with batching.
        """
        overall_start_time = time.time()
        
        path_objects = [Path(p) for p in file_paths if Path(p).is_file()]
        
        # 1. Deduplicate files before processing
        files_to_process = self._filter_new_files(path_objects)
        if not files_to_process:
            self.logger.info("No new documents to index.")
            return

        self.logger.info(f"Found {len(files_to_process)} new documents to index...")

        # 2. Process files in batches using ThreadPoolExecutor
        all_docs = []
        
        # Process files in batches to control memory usage
        for i in range(0, len(files_to_process), self.batch_size):
            batch_start_time = time.time()
            batch_files = files_to_process[i:i + self.batch_size]
            batch_num = i//self.batch_size + 1
            total_batches = (len(files_to_process) + self.batch_size - 1)//self.batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            batch_docs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit tasks for current batch
                future_to_file = {
                    executor.submit(self._load_and_prep_document, fp, thread_id): fp 
                    for fp in batch_files
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    fp = future_to_file[future]
                    try:
                        docs = future.result()
                        if docs: 
                            batch_docs.extend(docs)
                    except Exception as exc:
                        self.logger.error(f"'{fp.name}' generated an exception: {exc}")
            
            batch_load_time = time.time() - batch_start_time
            self.logger.info(f"  ⏱️ Batch {batch_num} document loading time: {batch_load_time:.3f}s")
            
            # 3. Add current batch to the appropriate store
            if batch_docs:
                upload_start_time = time.time()
                self._add_documents_batch(batch_docs)
                upload_time = time.time() - upload_start_time
                self.logger.info(f"  ⏱️ Batch {batch_num} upload time: {upload_time:.3f}s")
                all_docs.extend(batch_docs)
        
        overall_time = time.time() - overall_start_time
        self.logger.info(f"✅ Successfully indexed {len(files_to_process)} files with {len(all_docs)} total chunks in batches of {self.batch_size}. Total time: {overall_time:.3f}s")

    def _add_documents_batch(self, documents: List[Document]):
        """Add documents in batches for optimal performance."""
        if self.strategy == "basic":
            # For basic strategy, add chunks directly to vector store in batches
            vector_store = self._ensure_vector_store()
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                try:
                    # Time embedding separately
                    embed_start_time = time.time()
                    # Pre-compute embeddings
                    texts = [doc.page_content for doc in batch]
                    embeddings = self.cached_embedder.embed_documents(texts)
                    embed_time = time.time() - embed_start_time
                    
                    # Time upload separately
                    upload_start_time = time.time()
                    vector_store.add_documents(batch)
                    upload_time = time.time() - upload_start_time
                    
                    total_time = embed_time + upload_time
                    self.logger.info(f"    📦 Batch of {len(batch)} docs: embedding={embed_time:.3f}s, upload={upload_time:.3f}s, total={total_time:.3f}s")
                except Exception as e:
                    self.logger.error(f"    ❌ Error adding batch to vector store: {e}")
        else:
            # For hierarchical, use ParentDocumentRetriever (handles its own batching)
            retriever = self._ensure_retriever()
            try:
                # For hierarchical, we can't easily separate the timing since ParentDocumentRetriever
                # handles both embedding and upload internally
                embed_start_time = time.time()
                retriever.add_documents(documents, ids=None)
                total_time = time.time() - embed_start_time
                self.logger.info(f"    📦 Added {len(documents)} documents to hierarchical retriever (total time: {total_time:.3f}s)")
            except Exception as e:
                self.logger.error(f"    ❌ Error adding documents to hierarchical retriever: {e}")

    def index_directory(self, directory_path: str, thread_id: str):
        """Scans a directory and indexes all supported files in parallel."""
        directory = Path(directory_path)
        if not directory.is_dir():
            self.logger.error(f"Error: {directory_path} is not a valid directory.")
            return
        
        all_files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in self.LOADER_MAPPING]
        self.index_files([str(p) for p in all_files], thread_id)

    def _get_all_hashes_in_collection(self) -> set:
        """Get all file hashes that are already indexed in the collection."""
        try:
            # Check if collection exists first
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.vectors_count == 0:
                    return set()
            except Exception:
                # Collection doesn't exist yet
                return set()
            
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name, 
                limit=10000,
                with_payload=["metadata.file_hash"]
            )
            hashes = set()
            for point in scroll_result:
                if (point.payload and 
                    'metadata' in point.payload and 
                    'file_hash' in point.payload['metadata']):
                    hashes.add(point.payload['metadata']['file_hash'])
            return hashes
        except Exception as e:
            self.logger.warning(f"Warning: Could not retrieve existing hashes from collection: {e}")
            return set()

    def get_retriever(self, score_threshold: float = 0.3, k: int = 10, search_type: str = "similarity"):
        """Get a configured retriever based on the strategy."""
        if self.strategy == "basic":
            # Configure the vector store retriever
            vector_store = self._ensure_vector_store()
            return vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={
                    "k": k, 
                    "score_threshold": score_threshold
                }
            )
        else:
            # Configure the ParentDocumentRetriever
            retriever = self._ensure_retriever()
            retriever.search_kwargs = {
                'k': k, 
                'score_threshold': score_threshold
            }
            return retriever

    def search_documents(self, query: str, k: int = 5, return_metadata: bool = True) -> List[dict]:
        """
        Advanced search with multiple options and better control over results.
        Returns properly sized chunks regardless of strategy.
        """
        search_start_time = time.time()
        
        # Time query embedding
        embed_start_time = time.time()
        query_embedding = self.cached_embedder.embed_query(query)
        embed_time = time.time() - embed_start_time
        self.logger.info(f"🔍 Query embedding time: {embed_time:.3f}s")
        
        # Time search operation
        search_op_start_time = time.time()
        if self.strategy == "basic":
            # Direct vector search - already optimal chunk sizes
            vector_store = self._ensure_vector_store()
            docs = vector_store.similarity_search(query, k=k)
        else:
            # Use the retriever but potentially re-chunk large results
            retriever = self._ensure_retriever()
            docs = retriever.invoke(query)[:k]  # Limit results
            
            # If documents are too large, re-chunk them
            final_docs = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            for doc in docs:
                if len(doc.page_content) > self.chunk_size * 1.5:
                    # Split large documents into smaller chunks
                    chunks = splitter.split_documents([doc])
                    # Find the most relevant chunk
                    chunk_scores = []
                    for chunk in chunks:
                        # Simple relevance scoring (you could use embeddings here)
                        score = sum(1 for word in query.lower().split() 
                                  if word in chunk.page_content.lower())
                        chunk_scores.append((chunk, score))
                    
                    # Sort by relevance and take the best chunks
                    chunk_scores.sort(key=lambda x: x[1], reverse=True)
                    final_docs.extend([chunk for chunk, _ in chunk_scores[:2]])  # Top 2 chunks
                else:
                    final_docs.append(doc)
            docs = final_docs[:k]
        
        search_op_time = time.time() - search_op_start_time
        self.logger.info(f"🔍 Search operation time: {search_op_time:.3f}s")
        
        # Format results
        results = []
        for i, doc in enumerate(docs):
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "content_length": len(doc.page_content)
            }
            if return_metadata:
                result["metadata"] = doc.metadata
            results.append(result)
        
        total_search_time = time.time() - search_start_time
        self.logger.info(f"🔍 Total retrieval time: {total_search_time:.3f}s (returned {len(results)} results)")
            
        return results

    def clear_cache(self):
        """Clear the embedding cache directory."""
        try:
            import shutil
            cache_dir = Path(self.cache_path)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                os.makedirs(self.cache_path, exist_ok=True)
                self.logger.info(f"✅ Cleared cache directory: {self.cache_path}")
            else:
                self.logger.info(f"Cache directory does not exist: {self.cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")

    def delete_collection_if_exists(self):
        """Delete the collection if it exists."""
        try:
            if self._collection_exists():
                self.client.delete_collection(collection_name=self.collection_name)
                self.logger.info(f"✅ Deleted existing collection '{self.collection_name}'")
            else:
                self.logger.info(f"Collection '{self.collection_name}' does not exist, nothing to delete")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self.logger.info(f"Collection '{self.collection_name}' has been deleted.")

    def _delete_by_filter(self, filter_model: models.Filter):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=filter_model),
        )

    def delete_by_thread_id(self, thread_id: str):
        self._delete_by_filter(
            models.Filter(must=[models.FieldCondition(key="metadata.thread_id", match=models.MatchValue(value=thread_id))])
        )
        self.logger.info(f"Deleted all documents with thread_id: {thread_id}")


