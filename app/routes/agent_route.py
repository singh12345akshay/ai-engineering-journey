from fastapi import APIRouter, HTTPException
from app.models import AgentQuery, AgentResponse
from app.services.agent_service import run_agent
from app.models import AgentQuery, AgentResponse, GraphQuery, GraphResponse
from app.services.agent_service import run_agent
from app.services.langgraph_service import run_graph
from app.models import AgentQuery, AgentResponse, GraphQuery, GraphResponse, ResearchQuery, ResearchResponse
from app.services.agent_service import run_agent
from app.services.langgraph_service import run_graph
from app.services.multi_agent_service import run_research_pipeline

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
    
@router.post("/graph", response_model=GraphResponse)
async def graph_agent(query: GraphQuery):
    """
    LangGraph agent — explicit state machine with conditional routing.
    Classifies query then routes to document search or direct answer.
    More predictable than ReAct agent.
    """
    try:
        result = await run_graph(query.question)
        return GraphResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/research", response_model=ResearchResponse)
async def research_agent(query: ResearchQuery):
    """
    Multi-agent research pipeline.
    3 specialized agents: Researcher → Writer → Critic
    Each agent has one focused job.
    Takes 30-60 seconds — runs 3 LLM calls in sequence.
    """
    try:
        result = await run_research_pipeline(query.question)
        return ResearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))