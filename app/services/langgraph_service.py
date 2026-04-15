"""
LangGraph agent service.
Uses explicit state machine instead of ReAct loop.
You control the flow — not the LLM.
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.services.embeddings import search_documents
import operator


# ── State Definition ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state passed between every node in the graph.
    Every node reads from this and writes back to it.
    """
    question: str                    # the user's question
    query_type: str                  # "document" or "general"
    retrieved_docs: list             # chunks from ChromaDB
    context: str                     # formatted context string
    answer: str                      # final answer
    sources: list                    # source citations


# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.7
)

parser = StrOutputParser()


# ── Node 1: Classify Query ───────────────────────────────────────────────────

classify_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier. Classify the user's question into one of two categories:
- "document": questions about company policies, leave rules, internal documents, HR information
- "general": everything else — general knowledge, math, current events, coding questions

Respond with ONLY the word "document" or "general". Nothing else."""),
    ("human", "{question}")
])

classify_chain = classify_prompt | llm | parser


async def classify_query(state: AgentState) -> AgentState:
    """
    Node 1: Classify whether question needs document search or direct answer.
    Updates state with query_type.
    """
    print(f"\n[Node: classify_query] Question: {state['question']}")

    result = await classify_chain.ainvoke({"question": state["question"]})
    query_type = result.strip().lower()

    # safety check — default to document if unclear
    if query_type not in ["document", "general"]:
        query_type = "document"

    print(f"[Node: classify_query] Type: {query_type}")

    return {**state, "query_type": query_type}


# ── Node 2: Retrieve Documents ───────────────────────────────────────────────

async def retrieve_documents(state: AgentState) -> AgentState:
    """
    Node 2: Search ChromaDB for relevant document chunks.
    Only runs if query_type is 'document'.
    Updates state with retrieved_docs and context.
    """
    print(f"\n[Node: retrieve_documents] Searching for: {state['question']}")

    docs = search_documents(query=state["question"], n_results=3)

    if not docs:
        return {
            **state,
            "retrieved_docs": [],
            "context": "No relevant documents found.",
            "sources": []
        }

    # format context
    context_parts = []
    sources = []

    for doc in docs:
        source = doc["metadata"].get("source", "unknown")
        page = doc["metadata"].get("page_or_slide", "?")
        context_parts.append(f"[{source} p.{page}]: {doc['text']}")
        sources.append(f"{source} — page {page}")

    context = "\n\n".join(context_parts)

    print(f"[Node: retrieve_documents] Found {len(docs)} chunks")

    return {
        **state,
        "retrieved_docs": docs,
        "context": context,
        "sources": list(set(sources))
    }


# ── Node 3: Generate Answer from Documents ───────────────────────────────────

rag_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context say: "I don't have that information in the provided documents."
Always cite the source document and page number.

Context:
{context}"""),
    ("human", "{question}")
])

rag_answer_chain = rag_answer_prompt | llm | parser


async def generate_answer(state: AgentState) -> AgentState:
    """
    Node 3: Generate answer using retrieved document context.
    Runs after retrieve_documents for document queries.
    """
    print(f"\n[Node: generate_answer] Generating RAG answer")

    answer = await rag_answer_chain.ainvoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {**state, "answer": answer}


# ── Node 4: Generate Direct Answer ───────────────────────────────────────────

direct_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the question clearly and concisely."),
    ("human", "{question}")
])

direct_answer_chain = direct_answer_prompt | llm | parser


async def generate_direct(state: AgentState) -> AgentState:
    """
    Node 4: Generate direct answer without document retrieval.
    Runs for general knowledge questions.
    """
    print(f"\n[Node: generate_direct] Generating direct answer")

    answer = await direct_answer_chain.ainvoke({
        "question": state["question"]
    })

    return {**state, "answer": answer, "sources": []}


# ── Conditional Edge ─────────────────────────────────────────────────────────

def route_query(state: AgentState) -> str:
    """
    Conditional edge — decides which node to run next
    based on the query_type in state.

    This is YOU controlling the flow — not the LLM.
    """
    if state["query_type"] == "document":
        return "retrieve_documents"
    else:
        return "generate_direct"


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LangGraph state machine.

    Graph structure:
    START → classify_query → [conditional routing]
                              ├── document → retrieve_documents → generate_answer → END
                              └── general  → generate_direct → END
    """
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_direct", generate_direct)

    # set entry point
    graph.set_entry_point("classify_query")

    # add conditional edge after classification
    graph.add_conditional_edges(
        "classify_query",      # from this node
        route_query,           # use this function to decide
        {
            "retrieve_documents": "retrieve_documents",  # if returns "retrieve_documents"
            "generate_direct": "generate_direct"         # if returns "generate_direct"
        }
    )

    # add fixed edges
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("generate_direct", END)

    return graph.compile()


# compile graph once at startup
graph = build_graph()


async def run_graph(question: str) -> dict:
    """
    Run the LangGraph agent on a question.
    Returns answer with sources and query type.
    """
    initial_state = AgentState(
        question=question,
        query_type="",
        retrieved_docs=[],
        context="",
        answer="",
        sources=[]
    )

    result = await graph.ainvoke(initial_state)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "query_type": result["query_type"],
        "sources": result["sources"],
        "chunks_used": len(result["retrieved_docs"])
    }