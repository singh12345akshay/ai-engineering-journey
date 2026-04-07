from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models import DocumentInput, SearchQuery, SearchResponse, SearchResult
from app.services.embeddings import add_document, search_documents, get_collection_count

import tempfile
import os
from app.services.document_parser import parse_and_chunk_document
from app.services.embeddings import (
    add_document, search_documents,
    get_collection_count, add_document_chunks
)
from app.models import (
    DocumentInput, SearchQuery, SearchResponse,
    SearchResult, DocumentUploadResponse
)

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
    
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX file.
    Extracts text, chunks it, embeds it, stores in ChromaDB.
    """
    # validate file type
    allowed = [".pdf", ".pptx", ".ppt"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use PDF or PPTX."
        )

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # parse and chunk the document
        chunks = parse_and_chunk_document(tmp_path,original_filename=file.filename)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from this file."
            )

        # use filename without extension as doc ID prefix
        doc_id_prefix = os.path.splitext(file.filename)[0]

        # embed and store all chunks
        result = add_document_chunks(chunks, doc_id_prefix)

        return DocumentUploadResponse(
            filename=file.filename,
            chunks_added=result["chunks_added"],
            status="success",
            message=f"Extracted and indexed {result['chunks_added']} chunks from {file.filename}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # always clean up temp file
        os.unlink(tmp_path)