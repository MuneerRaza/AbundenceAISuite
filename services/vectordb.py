import time, hashlib, logging
from typing import List, Dict
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
load_dotenv()


from config import USER_CHROMA_PATH, USER_COLLECTION, VECTOR_SEARCH_K

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO
)

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
embedder = NVIDIAEmbeddings(model="baai/bge-m3")

# embedder = HuggingFaceEmbeddings(model="BAAI/bge-m3")
chroma = Chroma(
    persist_directory=USER_CHROMA_PATH,
    collection_name=USER_COLLECTION,
    embedding_function=embedder
)
reranker_ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L4-v2")
reranker = CrossEncoderReranker(model=reranker_ce, top_n=3)

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def load_paths(paths: List[str], thread_id: str) -> List[Dict]:
    metas = []
    for p in paths:
        metas.append({
            "path": p,
            "thread_id": thread_id,
            "file_hash": file_hash(p)
        })
    return metas

def ingest(metas: List[Dict]):
    for m in metas:
        exists = chroma._collection.get(
            where={"$and": [{"thread_id": m["thread_id"]}, {"file_hash": m["file_hash"]}]}
        )
        if exists and exists.get("documents"):
            logging.info(f"SKIP {m['path']}: already processed")
            continue
        ld_start = time.time()
        try:
            elements = partition(filename=m["path"], strategy="fast", languages=['en'])
            chunks_as_elements = chunk_by_title(
                elements,
                max_characters=512,
                combine_text_under_n_chars=128
            )
        except Exception as e:
            logging.error(f"Failed to load {m['path']} with Unstructured: {e}")
            continue 
        logging.info(f"Unstructured load time: {time.time() - ld_start:.2f}s for {m['path']}")
        sp_start = time.time()

        chunks = []
        for element in chunks_as_elements:
            metadata = element.metadata.to_dict()
            
            metadata.update(m)
            chunks.append(Document(page_content=element.text, metadata=metadata))
        chunks = filter_complex_metadata(chunks)
        logging.info(f"Text splitting time: {time.time() - sp_start:.2f}s for {m['path']} → {len(chunks)} chunks")
        
        for chunk in chunks:
            chunk.metadata.update(m)
        
        if not chunks:
            logging.warning(f"No text chunks extracted from {m['path']}. Skipping ingestion.")
            continue
        start = time.time()
        chroma.add_documents(chunks)
        logging.info(f"Ingested {m['path']} → {len(chunks)} chunks in {time.time()-start:.2f}s")

def advanced_retrieve(query: str, thread_id: str) -> List[Document]:
    docs_for_thread = chroma.get(
        where={"thread_id": thread_id},
        include=["documents", "metadatas"]
    )
    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs_for_thread.get("documents", []), docs_for_thread.get("metadatas", []))
    ]

    if not documents:
        logging.warning(f"No documents found for thread_id: {thread_id}")
        return []
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = VECTOR_SEARCH_K

    dense_retriever = chroma.as_retriever(
        search_kwargs={
            "k": VECTOR_SEARCH_K,
            "filter": {"thread_id": thread_id}
        }
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5] 
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=reranker
    )

    t0 = time.time()
    docs = compression_retriever.invoke(query)
    logging.info(f"Query '{query}' on thread_id '{thread_id}' → {len(docs)} docs in {time.time() - t0:.2f}s")
    return docs

def main(paths: List[str], query: str, thread_id: str):
    metas = load_paths(paths, thread_id)
    ingest(metas)
    docs = advanced_retrieve(query, thread_id)
    for d in docs:
        print(f"---\n{d.metadata}\n{d.page_content}\n")

if __name__ == "__main__":
    # This example will now use the new Unstructured-based ingestion
    file_list = [r"C:\Users\MuneerRaza\Downloads\CognifootAI_FYP_report_final.pdf"]
    query = "What is the accuracy of latent classifier they acheived?"
    thread_id = "b"
    main(file_list, query, thread_id)