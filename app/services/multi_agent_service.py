"""
Multi-agent research pipeline using LangGraph.
3 specialized agents: Researcher → Writer → Critic
Each agent has one focused job and passes results via shared state.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from app.config import settings
from app.services.embeddings import search_documents


# ── State ────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """
    Shared state passed between all 3 agents.
    Each agent reads previous results and adds its own.
    """
    question: str           # original user question
    research_notes: str     # researcher's raw findings
    sources_used: list      # sources the researcher found
    draft_answer: str       # writer's synthesized answer
    critique: str           # critic's feedback
    final_answer: str       # final polished answer
    quality_score: int      # critic's quality score 1-10


# ── Shared LLM ───────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3
)

parser = StrOutputParser()
web_search = DuckDuckGoSearchRun()


# ── Agent 1: Researcher ──────────────────────────────────────────────────────

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research specialist. Your job is to find accurate,
relevant information to answer a question.

You have access to:
1. Web search results provided to you
2. Internal document search results provided to you

Your output must be structured as:
FINDINGS:
[bullet points of key facts you found]

SOURCES:
[list the sources where you found information]

Be thorough. Include specific numbers, dates, and details.
Do not write a final answer — just research notes."""),
    ("human", """Question: {question}

Web Search Results:
{web_results}

Internal Document Results:
{doc_results}

Now write your research notes:""")
])

researcher_chain = researcher_prompt | llm | parser


async def researcher_agent(state: ResearchState) -> ResearchState:
    """
    Agent 1: Researcher
    Searches web and internal documents, extracts relevant facts.
    """
    print(f"\n[Agent 1: Researcher] Researching: {state['question'][:50]}...")

    # search web
    try:
        web_results = web_search.run(state["question"])
    except Exception:
        web_results = "Web search unavailable."

    # search internal documents
    doc_results_raw = search_documents(query=state["question"], n_results=3)
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

    # run researcher
    research_notes = await researcher_chain.ainvoke({
        "question": state["question"],
        "web_results": web_results,
        "doc_results": doc_results
    })

    print(f"[Agent 1: Researcher] Research complete. Found {len(sources)} sources.")

    return {
        **state,
        "research_notes": research_notes,
        "sources_used": sources
    }


# ── Agent 2: Writer ──────────────────────────────────────────────────────────

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical writer. Your job is to take research notes
and synthesize them into a clear, well-structured answer.

Rules:
- Write in clear, simple language
- Structure the answer with a direct answer first, then details
- Include specific facts and numbers from the research
- Keep it concise — no fluff
- Do not add information not in the research notes"""),
    ("human", """Question: {question}

Research Notes:
{research_notes}

Write a clear, structured answer:""")
])

writer_chain = writer_prompt | llm | parser


async def writer_agent(state: ResearchState) -> ResearchState:
    """
    Agent 2: Writer
    Synthesizes research notes into a clear structured answer.
    """
    print(f"\n[Agent 2: Writer] Writing answer from research notes...")

    draft = await writer_chain.ainvoke({
        "question": state["question"],
        "research_notes": state["research_notes"]
    })

    print(f"[Agent 2: Writer] Draft complete.")

    return {**state, "draft_answer": draft}


# ── Agent 3: Critic ──────────────────────────────────────────────────────────

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a quality reviewer. Your job is to review an answer
and assess its quality.

Evaluate the answer on:
1. Accuracy — does it match the research notes?
2. Completeness — does it fully answer the question?
3. Clarity — is it easy to understand?
4. Conciseness — is it appropriately brief?

Your output must be structured EXACTLY as:
SCORE: [number 1-10]
ISSUES: [list any problems, or "None" if no issues]
IMPROVED_ANSWER: [the answer with any improvements applied]"""),
    ("human", """Question: {question}

Research Notes:
{research_notes}

Draft Answer:
{draft_answer}

Review this answer:""")
])

critic_chain = critic_prompt | llm | parser


async def critic_agent(state: ResearchState) -> ResearchState:
    """
    Agent 3: Critic
    Reviews the draft answer, scores it, and improves it if needed.
    """
    print(f"\n[Agent 3: Critic] Reviewing draft answer...")

    critique = await critic_chain.ainvoke({
        "question": state["question"],
        "research_notes": state["research_notes"],
        "draft_answer": state["draft_answer"]
    })

    # extract score from critique
    quality_score = 7  # default
    for line in critique.split("\n"):
        if line.startswith("SCORE:"):
            try:
                quality_score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                pass

    # extract improved answer
    final_answer = state["draft_answer"]  # fallback to draft
    if "IMPROVED_ANSWER:" in critique:
        improved = critique.split("IMPROVED_ANSWER:")[-1].strip()
        if improved and len(improved) > 50:
            final_answer = improved

    print(f"[Agent 3: Critic] Review complete. Quality score: {quality_score}/10")

    return {
        **state,
        "critique": critique,
        "final_answer": final_answer,
        "quality_score": quality_score
    }


# ── Build Graph ──────────────────────────────────────────────────────────────

def build_research_graph():
    """
    Build the 3-agent research pipeline graph.

    Flow: START → researcher → writer → critic → END

    Linear pipeline — each agent builds on the previous one's work.
    No conditional edges needed — all questions go through all 3 agents.
    """
    graph = StateGraph(ResearchState)

    # add all 3 agent nodes
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)

    # set entry point
    graph.set_entry_point("researcher")

    # linear flow — researcher → writer → critic → end
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


research_graph = build_research_graph()


async def run_research_pipeline(question: str) -> dict:
    """
    Run the full 3-agent research pipeline.
    Returns final answer with quality score and sources.
    """
    initial_state = ResearchState(
        question=question,
        research_notes="",
        sources_used=[],
        draft_answer="",
        critique="",
        final_answer="",
        quality_score=0
    )

    print(f"\n{'='*50}")
    print(f"Starting multi-agent research pipeline")
    print(f"Question: {question}")
    print(f"{'='*50}")

    result = await research_graph.ainvoke(initial_state)

    print(f"\n{'='*50}")
    print(f"Pipeline complete. Quality score: {result['quality_score']}/10")
    print(f"{'='*50}\n")

    return {
        "question": result["question"],
        "answer": result["final_answer"],
        "quality_score": result["quality_score"],
        "sources": result["sources_used"],
        "draft_answer": result["draft_answer"],
        "critique": result["critique"]
    }