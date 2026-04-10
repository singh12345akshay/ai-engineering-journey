from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.embeddings import collection, embed_text
from app.config import settings
import numpy as np

# embedding model — same as embeddings.py
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# cross-encoder for re-ranking
# this model scores query-document pairs more accurately than cosine similarity
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def get_all_documents() -> list[dict]:
    """
    Fetch all documents from ChromaDB for BM25 indexing.
    BM25 needs the full corpus to calculate term frequencies.
    """
    results = collection.get()

    if not results['ids']:
        return []

    documents = []
    for i in range(len(results['ids'])):
        documents.append({
            "id": results['ids'][i],
            "text": results['documents'][i],
            "metadata": results['metadatas'][i]
        })

    return documents


def bm25_search(query: str, documents: list[dict],
                n_results: int = 10) -> list[dict]:
    """
    BM25 keyword search over documents.
    Returns top n_results ranked by keyword relevance.
    """
    if not documents:
        return []

    # tokenize — split into words for BM25
    tokenized_corpus = [doc["text"].lower().split() for doc in documents]
    tokenized_query = query.lower().split()

    # build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # get scores for all documents
    scores = bm25.get_scores(tokenized_query)

    # pair documents with scores and sort
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # return top n with scores
    results = []
    for doc, score in scored_docs[:n_results]:
        if score > 0:  # skip documents with zero BM25 score
            results.append({**doc, "bm25_score": float(score)})

    return results


def semantic_search(query: str, n_results: int = 10) -> list[dict]:
    """
    Semantic search using ChromaDB cosine similarity.
    Returns top n_results ranked by semantic similarity.
    """
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    if not results['ids'][0]:
        return []

    documents = []
    for i in range(len(results['ids'][0])):
        similarity = round(1 - results['distances'][0][i], 4)
        documents.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "semantic_score": similarity
        })

    return documents


def hybrid_search(query: str, n_results: int = 10,
                  alpha: float = 0.5) -> list[dict]:
    """
    Hybrid search — combines BM25 keyword + semantic similarity.

    alpha controls the balance:
    alpha=0.0 → pure BM25 keyword search
    alpha=0.5 → equal weight (default)
    alpha=1.0 → pure semantic search

    Both scores are normalised to 0-1 before combining.
    """
    all_docs = get_all_documents()

    if not all_docs:
        return []

    # run both searches
    bm25_results = bm25_search(query, all_docs, n_results=n_results)
    semantic_results = semantic_search(query, n_results=n_results)

    # build lookup dicts by doc ID
    bm25_map = {r["id"]: r["bm25_score"] for r in bm25_results}
    semantic_map = {r["id"]: r["semantic_score"] for r in semantic_results}

    # get all unique doc IDs from both searches
    all_ids = set(bm25_map.keys()) | set(semantic_map.keys())

    # normalise BM25 scores to 0-1 range
    bm25_max = max(bm25_map.values()) if bm25_map else 1
    bm25_normalised = {
        id: score / bm25_max
        for id, score in bm25_map.items()
    }

    # combine scores
    combined = []
    for doc_id in all_ids:
        bm25_score = bm25_normalised.get(doc_id, 0)
        semantic_score = semantic_map.get(doc_id, 0)

        # weighted combination
        hybrid_score = (1 - alpha) * bm25_score + alpha * semantic_score

        # get document text and metadata
        if doc_id in semantic_map:
            doc = next(d for d in semantic_results if d["id"] == doc_id)
        else:
            doc = next(d for d in bm25_results if d["id"] == doc_id)

        combined.append({
            "id": doc_id,
            "text": doc["text"],
            "metadata": doc["metadata"],
            "bm25_score": round(bm25_map.get(doc_id, 0), 4),
            "semantic_score": round(semantic_map.get(doc_id, 0), 4),
            "hybrid_score": round(hybrid_score, 4)
        })

    # sort by hybrid score
    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return combined[:n_results]


def rerank_documents(query: str, documents: list[dict],
                     top_k: int = 3) -> list[dict]:
    """
    Re-rank documents using a cross-encoder model.

    Cross-encoder scores query-document pairs together — much more
    accurate than cosine similarity which scores them independently.

    Two-stage pipeline:
    Stage 1: hybrid_search retrieves 10 candidates (fast)
    Stage 2: rerank_documents picks best 3 from 10 (accurate)
    """
    if not documents:
        return []

    # prepare query-document pairs for cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]

    # score all pairs
    scores = reranker.predict(pairs)

    # attach scores to documents
    for i, doc in enumerate(documents):
        doc["rerank_score"] = float(scores[i])

    # sort by rerank score
    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]


async def advanced_rag_search(query: str,
                               n_candidates: int = 10,
                               final_results: int = 3,
                               alpha: float = 0.5) -> list[dict]:
    """
    Full advanced RAG retrieval pipeline:
    1. Hybrid search — retrieve n_candidates using BM25 + semantic
    2. Re-rank — pick final_results from candidates using cross-encoder

    This two-stage approach gives speed (hybrid) and accuracy (reranker).
    """
    # stage 1 — hybrid search for candidates
    candidates = hybrid_search(query, n_results=n_candidates, alpha=alpha)

    if not candidates:
        return []

    # stage 2 — re-rank candidates
    reranked = rerank_documents(query, candidates, top_k=final_results)

    return reranked