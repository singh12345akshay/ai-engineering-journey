from fastapi import APIRouter, HTTPException
from app.models import LangChainQuery, LangChainResponse, ConversationHistory
from app.services.langchain_service import (
    ask_simple,
    ask_with_memory,
    clear_memory,
    get_conversation_history
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