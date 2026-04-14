from fastapi import APIRouter, HTTPException
from app.models import LangChainQuery, LangChainResponse, ConversationHistory
from app.services.langchain_service import (
    ask_simple,
    ask_with_memory,
    clear_memory,
    get_conversation_history
)
from app.models import (
    RAGQuery, RAGResponse
)
from app.config import settings  
from app.services.langchain_service import (
    ask_with_rag,
    ask_conversational_rag
)
from app.models import (
    LangChainQuery, LangChainResponse,
    ConversationHistory, RAGQuery, RAGResponse,
    AdvancedRAGQuery, DocumentQARequest, DocumentQAResponse
)
from app.services.langchain_service import (
    ask_simple,
    ask_with_memory,
    ask_with_rag,
    ask_conversational_rag,
    ask_advanced_rag,
    document_qa,
    clear_memory,
    get_conversation_history
)
from app.services.langchain_service import (
    ask_simple,
    ask_with_memory,
    ask_with_rag,
    ask_conversational_rag,
    ask_advanced_rag,
    clear_memory,
    get_conversation_history
)
from app.models import (
    LangChainQuery, LangChainResponse,
    ConversationHistory, RAGQuery, RAGResponse,
    AdvancedRAGQuery
)

router = APIRouter(prefix="/lc", tags=["LangChain"])


@router.get("/health")
async def langchain_health():
    """Check LangChain service is running."""
    return {
        "status": "running",
        "service": "LangChain",
        "features": ["simple_qa", "conversation_memory"]
    }


@router.post("/ask", response_model=LangChainResponse)
async def ask(query: LangChainQuery):
    """
    Simple QA using LangChain chain.
    Same as /ai/ask but using LCEL pipeline.
    """
    try:
        result = await ask_simple(query.question)
        return LangChainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=LangChainResponse)
async def chat(query: LangChainQuery):
    """
    Conversational QA with memory.
    Send same session_id to continue a conversation.
    Ask follow-up questions — the model remembers context.
    """
    try:
        result = await ask_with_memory(query.question, query.session_id)
        return LangChainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=ConversationHistory)
async def get_history(session_id: str):
    """Get full conversation history for a session."""
    return get_conversation_history(session_id)


@router.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """Clear conversation history for a session."""
    return clear_memory(session_id)


@router.post("/rag", response_model=RAGResponse)
async def rag(query: RAGQuery):
    """
    RAG endpoint — retrieves from your uploaded documents
    then generates a grounded answer with source citations.
    No memory — each call is independent.
    """
    try:
        result = await ask_with_rag(
            question=query.question,
            n_results=query.n_results
        )
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chat", response_model=RAGResponse)
async def rag_chat(query: RAGQuery):
    """
    Conversational RAG — retrieval + generation + memory.
    The complete production RAG pattern.
    Send same session_id to maintain conversation context.
    """
    try:
        result = await ask_conversational_rag(
            question=query.question,
            session_id=query.session_id,
            n_results=query.n_results
        )
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/rag/advanced", response_model=RAGResponse)
async def advanced_rag(query: AdvancedRAGQuery):
    """
    Advanced RAG — hybrid search + re-ranking + generation + memory.

    alpha parameter controls BM25 vs semantic balance:
    0.0 = pure keyword, 0.5 = balanced, 1.0 = pure semantic

    Two-stage retrieval:
    Stage 1: hybrid search retrieves n_candidates
    Stage 2: cross-encoder re-ranks to final_results
    """
    try:
        result = await ask_advanced_rag(
            question=query.question,
            session_id=query.session_id,
            n_candidates=query.n_candidates,
            final_results=query.final_results
        )
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from groq import RateLimitError

@router.post("/document/qa", response_model=DocumentQAResponse)
async def production_document_qa(request: DocumentQARequest):
    try:
        result = await document_qa(request.model_dump())
        return DocumentQAResponse(**result)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="LLM rate limit reached. Please wait a few minutes and try again."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/system/status")
async def system_status():
    """
    Complete system status — shows everything that's running.
    Useful for monitoring and debugging in production.
    """
    from app.services.embeddings import get_collection_count
    from app.services.langchain_service import conversation_memories

    doc_count = get_collection_count()
    active_sessions = len(conversation_memories)

    return {
        "status": "running",
        "version": settings.APP_VERSION,
        "model": settings.MODEL,
        "documents_indexed": doc_count,
        "active_sessions": active_sessions,
        "langsmith_tracing": settings.LANGCHAIN_TRACING_V2 == "true",
        "endpoints": {
            "simple_qa": "POST /lc/ask",
            "chat_with_memory": "POST /lc/chat",
            "basic_rag": "POST /lc/rag",
            "conversational_rag": "POST /lc/rag/chat",
            "advanced_rag": "POST /lc/rag/advanced",
            "production_qa": "POST /lc/document/qa",
            "document_upload": "POST /search/upload",
            "semantic_search": "POST /search/query"
        },
        "rag_settings": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "default_alpha": settings.DEFAULT_ALPHA,
            "min_similarity": settings.MIN_SIMILARITY_THRESHOLD
        }
    }