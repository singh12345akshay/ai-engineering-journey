from fastapi import APIRouter, HTTPException
from app.models import DocumentInput, SearchQuery, SearchResponse, SearchResult
from app.services.embeddings import add_document, search_documents, get_collection_count

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/health")
async def search_health():
    count = get_collection_count()
    return {
        "status": "running",
        "documents_indexed": count,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "ChromaDB"
    }


@router.post("/add")
async def add_doc(document: DocumentInput):
    try:
        result = add_document(
            doc_id=document.doc_id,
            text=document.text,
            metadata=document.metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=SearchResponse)
async def search(query: SearchQuery):
    try:
        results = search_documents(
            query=query.query,
            n_results=query.n_results
        )

        return SearchResponse(
            query=query.query,
            results=[SearchResult(**r) for r in results],
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))