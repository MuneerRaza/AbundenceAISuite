import time, hashlib, os, pickle
import concurrent.futures
from typing import List, Dict
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
load_dotenv()


from config import CACHE_DIR, USER_CHROMA_PATH, USER_COLLECTION, VECTOR_SEARCH_K

embedder = NVIDIAEmbeddings(model="baai/bge-m3")
# embedder = HuggingFaceEmbeddings(model="BAAI/bge-m3")
chroma = Chroma(
    persist_directory=USER_CHROMA_PATH,
    collection_name=USER_COLLECTION,
    embedding_function=embedder
)

def _ingest_single_file(meta: dict) -> int:
    """
    Helper function for true parallel ingestion.
    Handles the entire lifecycle of one file: read, chunk, embed, and add.
    """
    try:
        print(f"-> Processing {meta['path']}...")
        elements = partition(filename=meta["path"], strategy="fast")
        chunks_as_elements = chunk_by_title(
            elements, max_characters=1500, combine_text_under_n_chars=256
        )
        
        docs = []
        for element in chunks_as_elements:
            doc = Document(page_content=element.text, metadata=element.metadata.to_dict())
            doc.metadata.update(meta)
            docs.append(doc)
        
        if not docs:
            print(f"No text extracted from {meta['path']}. Skipping.")
            return 0
        
        docs = filter_complex_metadata(docs)

        chroma.add_documents(docs)
        
        return len(docs)
    except Exception as e:
        print(f"Error processing {meta['path']} in worker: {e}")
        return 0

def ingest_files(metas: List[dict]):
    """
    Ingests files with true end-to-end parallelism, including embedding.
    """
    metas_to_ingest = []
    for meta in metas:
        existing_docs = chroma._collection.get(where={"file_hash": meta["file_hash"]})
        if existing_docs and existing_docs.get("ids"):
             print(f"✅ Skipping '{os.path.basename(meta['path'])}'. Hash already exists in DB.")
             continue
        metas_to_ingest.append(meta)

    if not metas_to_ingest:
        print("--- All provided files have already been ingested. Nothing to do. ---")
        return
    
    print(f"---STARTING PARALLEL INGESTION FOR {len(metas)} FILES---")
    total_chunks_ingested = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_meta = {executor.submit(_ingest_single_file, meta): meta for meta in metas}
        
        for future in concurrent.futures.as_completed(future_to_meta):
            chunks_count = future.result()
            total_chunks_ingested += chunks_count

    print(f"✅ Ingestion complete. Ingested a total of {total_chunks_ingested} chunks from {len(metas)} files.")


def get_bm25_retriever(thread_id: str, documents: List[Document]):
    """
    Builds or loads a cached BM25 retriever to avoid re-computation.
    """
    cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{thread_id}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print(f"Loading cached BM25 index for thread '{thread_id}'")
            return pickle.load(f)
    else:
        print(f"Creating and caching new BM25 index for thread '{thread_id}'")
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = VECTOR_SEARCH_K
        with open(cache_path, "wb") as f:
            pickle.dump(bm25, f)
        return bm25

def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    """
    Merges multiple ranked lists of documents using RRF.
    """
    fused_scores = {}
    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = {"doc": doc, "score": 0}
            fused_scores[doc_str]["score"] += 1 / (rank + k)

    reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked_results]

def get_retrieved_docs(query: str, thread_id: str, strategy: str = "hybrid") -> List[Document]:
    """
    Performs hybrid retrieval using Vector Search and BM25, then fuses the results with RRF.
    This is the first stage before re-ranking.
    """
    docs_for_thread = chroma.get(
        where={"thread_id": thread_id}, include=["documents", "metadatas"]
    )
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs_for_thread.get("documents", []), docs_for_thread.get("metadatas", []))
    ]

    if not documents:
        print(f"No documents found for thread_id: {thread_id}")
        return []

    # --- NEW: Hybrid Search with RRF Logic ---
    t0 = time.time()
    retrieved_docs = []

    # 1. Vector Search (Dense)
    if strategy in ["vector_search", "hybrid"]:
        print("-> Performing dense (vector) search...")
        dense_retriever = chroma.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_K, "filter": {"thread_id": thread_id}}
        )
        retrieved_docs.append(dense_retriever.invoke(query))

    # 2. Keyword Search (Sparse)
    if strategy in ["keyword_search", "hybrid"]:
        print("-> Performing sparse (keyword) search...")
        bm25_retriever = get_bm25_retriever(thread_id, documents)
        retrieved_docs.append(bm25_retriever.invoke(query))

    # 3. Fuse results with RRF
    if len(retrieved_docs) > 1:
        print("-> Fusing results with RRF...")
        fused_docs = reciprocal_rank_fusion(retrieved_docs)
    else:
        fused_docs = retrieved_docs[0] if retrieved_docs else []

    print(f"Query '{query}' on thread '{thread_id}' -> {len(fused_docs)} docs in {time.time() - t0:.2f}s")
    return fused_docs[:VECTOR_SEARCH_K] # Return top K results after fusion

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def skip_if_exists(path: str, thread_id: str) -> bool:
    """
    Checks if a file with the same path and thread_id already exists in the collection.
    If it does, skips ingestion.
    """
    existing_docs = chroma._collection.get(where={
        "path": path,
        "thread_id": thread_id
    }, include=["metadatas"])
    
    if existing_docs and existing_docs.get("metadatas"):
        print(f"Skipping ingestion for {path} as it already exists in the collection.")
        return True
    return False

def load_paths(paths: List[str], thread_id: str) -> List[Dict]:
    metas = []
    for p in paths:
        metas.append({
            "path": p,
            "thread_id": thread_id,
            "file_hash": file_hash(p)
        })
    return metas

def delete_by_thread_id(thread_id: str):
    """Deletes all documents associated with a specific thread_id."""
    try:
        chroma._collection.delete(where={"thread_id": thread_id})
        print(f"✅ Successfully deleted all documents for thread_id: '{thread_id}'")
    except Exception as e:
        print(f"Error deleting documents for thread_id '{thread_id}': {e}")

def delete_by_hash(file_hash: str):
    """Deletes all documents associated with a specific file_hash."""
    try:
        chroma._collection.delete(where={"file_hash": file_hash})
        print(f"✅ Successfully deleted all documents for file_hash: '{file_hash}'")
    except Exception as e:
        print(f"Error deleting documents for file_hash '{file_hash}': {e}")

def delete_by_path(path: str):
    """Deletes all documents associated with a specific file path."""
    try:
        chroma._collection.delete(where={"path": path})
        print(f"✅ Successfully deleted all documents for path: '{path}'")
    except Exception as e:
        print(f"Error deleting documents for path '{path}': {e}")

def empty_bm25_cache(thread_id: str):
    """
    Deletes the cached BM25 index for a specific thread_id.
    """
    cache_dir = CACHE_DIR
    cache_path = os.path.join(cache_dir, f"{thread_id}.pkl")
    
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"✅ Successfully deleted BM25 cache for thread_id: '{thread_id}'")
    else:
        print(f"No BM25 cache found for thread_id: '{thread_id}'")

if __name__ == "__main__":
    import time
    
    # --- Test Setup ---
    # Make sure this path is correct for your system
    file_list = [r"C:\Users\MuneerRaza\Downloads\CognifootAI_FYP_report_final.pdf"]
    query = "What is the accuracy of latent classifier they acheived?"
    thread_id = "b"

    # --- Ingestion Timing ---
    print("\n--- TIMING INGESTION ---")
    ingest_start_time = time.time()

    metas = load_paths(file_list, thread_id)
    ingest_files(metas) # This now calls the parallel ingestor
    ingest_end_time = time.time()
    print(f"✅ Ingestion took: {ingest_end_time - ingest_start_time:.2f} seconds.")

    # --- Retrieval Timing ---
    print("\n--- TIMING RETRIEVAL ---")
    retrieve_start_time = time.time()
    # The retrieval function was renamed to get_retrieved_docs
    docs = get_retrieved_docs(query, thread_id)
    for doc in docs:
        print(f"Retrieved doc: {doc.page_content})")
    print(f"Retrieved {len(docs)} documents.")
    retrieve_end_time = time.time()
    print(f"✅ Retrieval took: {retrieve_end_time - retrieve_start_time:.2f} seconds.")

    total_time = (ingest_end_time - ingest_start_time) + (retrieve_end_time - retrieve_start_time)
    print(f"\nTotal script execution time: {total_time:.2f} seconds.")
