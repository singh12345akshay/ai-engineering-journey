from fastapi import APIRouter, HTTPException
from app.models import AgentQuery, AgentResponse
from app.services.agent_service import run_agent

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.get("/health")
async def agent_health():
    """Check agent service is running."""
    return {
        "status": "running",
        "pattern": "ReAct",
        "tools": [
            "web_search (DuckDuckGo)",
            "wikipedia",
            "calculator",
            "document_search (your RAG system)"
        ]
    }


@router.post("/ask", response_model=AgentResponse)
async def ask_agent(query: AgentQuery):
    """
    ReAct agent endpoint.
    Agent reasons about which tools to use and answers the question.

    Examples:
    - "What is the current population of India?" → uses web search
    - "Who invented the telephone?" → uses Wikipedia
    - "What is 15% of 2500?" → uses calculator
    - "How many casual leaves do I get?" → uses document search
    """
    try:
        result = await run_agent(query.question)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))