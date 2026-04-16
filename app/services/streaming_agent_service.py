"""
Streaming agent service.
Yields SSE events as agents work through their tasks.
Frontend sees progress in real time instead of waiting for completion.
"""
import json
import asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from app.config import settings
from app.services.embeddings import search_documents
from typing import AsyncGenerator


# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3
)
parser = StrOutputParser()
web_search = DuckDuckGoSearchRun()


# ── Prompts ───────────────────────────────────────────────────────────────────

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research specialist. Find accurate information.
Structure your output as:
FINDINGS:
[bullet points of key facts]
SOURCES:
[list sources]"""),
    ("human", """Question: {question}
Web Results: {web_results}
Document Results: {doc_results}
Write research notes:""")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer. Synthesize research into a clear structured answer. Be concise but complete."),
    ("human", """Question: {question}
Research Notes: {research_notes}
Write a clear answer:""")
])

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a quality reviewer. Review the answer and output EXACTLY:
SCORE: [1-10]
ISSUES: [issues or None]
IMPROVED_ANSWER: [improved version]"""),
    ("human", """Question: {question}
Research: {research_notes}
Draft: {draft_answer}
Review:""")
])


# ── SSE Event Helper ──────────────────────────────────────────────────────────

def make_event(event_type: str, data: dict) -> str:
    """
    Format a Server-Sent Event.
    event: identifies the type of event
    data: JSON payload

    Frontend listens for specific event types:
    - status: progress updates
    - result: intermediate results
    - done: final answer
    - error: something went wrong
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Streaming Pipeline ────────────────────────────────────────────────────────

async def stream_research_pipeline(question: str) -> AsyncGenerator[str, None]:
    """
    Stream the multi-agent research pipeline as SSE events.
    Yields events as each agent completes its work.
    """

    # ── Agent 1: Researcher ──────────────────────────────────────────────────

    yield make_event("status", {
        "agent": "Researcher",
        "message": "Starting research...",
        "step": 1,
        "total_steps": 3
    })

    await asyncio.sleep(0.1)  # small delay for frontend to render

    # web search
    yield make_event("status", {
        "agent": "Researcher",
        "message": "Searching the web...",
        "step": 1,
        "total_steps": 3
    })

    try:
        web_results = web_search.run(question)
    except Exception:
        web_results = "Web search unavailable."

    # document search
    yield make_event("status", {
        "agent": "Researcher",
        "message": "Searching internal documents...",
        "step": 1,
        "total_steps": 3
    })

    doc_results_raw = search_documents(query=question, n_results=3)
    if doc_results_raw:
        doc_results = "\n".join([
            f"[{d['metadata'].get('source', 'doc')} p.{d['metadata'].get('page_or_slide', '?')}]: {d['text']}"
            for d in doc_results_raw
        ])
        sources = list(set([
            f"{d['metadata'].get('source', 'unknown')} page {d['metadata'].get('page_or_slide', '?')}"
            for d in doc_results_raw
        ]))
    else:
        doc_results = "No relevant internal documents found."
        sources = []

    # run researcher LLM
    research_chain = researcher_prompt | llm | parser
    research_notes = await research_chain.ainvoke({
        "question": question,
        "web_results": web_results[:2000],  # truncate to save tokens
        "doc_results": doc_results
    })

    yield make_event("result", {
        "agent": "Researcher",
        "message": "Research complete",
        "sources_found": len(sources),
        "sources": sources,
        "step": 1,
        "total_steps": 3
    })

    # ── Agent 2: Writer ──────────────────────────────────────────────────────

    yield make_event("status", {
        "agent": "Writer",
        "message": "Writing answer from research notes...",
        "step": 2,
        "total_steps": 3
    })

    writer_chain = writer_prompt | llm | parser
    draft_answer = await writer_chain.ainvoke({
        "question": question,
        "research_notes": research_notes
    })

    yield make_event("result", {
        "agent": "Writer",
        "message": "Draft complete",
        "step": 2,
        "total_steps": 3
    })

    # ── Agent 3: Critic ──────────────────────────────────────────────────────

    yield make_event("status", {
        "agent": "Critic",
        "message": "Reviewing draft answer...",
        "step": 3,
        "total_steps": 3
    })

    critic_chain = critic_prompt | llm | parser
    critique = await critic_chain.ainvoke({
        "question": question,
        "research_notes": research_notes,
        "draft_answer": draft_answer
    })

    # extract score
    quality_score = 7
    for line in critique.split("\n"):
        if line.startswith("SCORE:"):
            try:
                quality_score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                pass

    # extract final answer
    final_answer = draft_answer
    if "IMPROVED_ANSWER:" in critique:
        improved = critique.split("IMPROVED_ANSWER:")[-1].strip()
        if improved and len(improved) > 50:
            final_answer = improved

    yield make_event("result", {
        "agent": "Critic",
        "message": f"Review complete. Quality score: {quality_score}/10",
        "quality_score": quality_score,
        "step": 3,
        "total_steps": 3
    })

    # ── Final Answer ─────────────────────────────────────────────────────────

    yield make_event("done", {
        "question": question,
        "answer": final_answer,
        "quality_score": quality_score,
        "sources": sources,
        "message": "Pipeline complete"
    })