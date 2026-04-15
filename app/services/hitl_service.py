"""
Human in the Loop service using LangGraph interrupts.
Agent proposes actions, human reviews and approves/rejects,
then pipeline continues or stops based on human decision.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
import json
import uuid


# ── State ─────────────────────────────────────────────────────────────────────

class HITLState(TypedDict):
    """
    State for human-in-the-loop pipeline.
    Tracks the full lifecycle from analysis to execution.
    """
    task_id: str                    # unique ID for this pipeline run
    content: str                    # content to process
    analysis: str                   # agent's analysis of the content
    proposed_actions: list          # list of actions agent wants to take
    human_decision: str             # "approved" or "rejected"
    human_feedback: str             # optional feedback from human
    execution_result: str           # result after execution
    status: str                     # current pipeline status


# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3
)
parser = StrOutputParser()


# ── Node 1: Analyze Content ───────────────────────────────────────────────────

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a document analyst. Analyze the provided content and
propose specific actions to take.

Your output must be valid JSON in exactly this format:
{{
    "analysis": "brief analysis of the content",
    "proposed_actions": [
        "action 1 description",
        "action 2 description",
        "action 3 description"
    ],
    "risk_level": "low/medium/high",
    "reasoning": "why these actions are appropriate"
}}

Return ONLY the JSON. No other text."""),
    ("human", "Analyze this content and propose actions:\n\n{content}")
])

analyze_chain = analyze_prompt | llm | parser


async def analyze_document(state: HITLState) -> HITLState:
    """
    Node 1: Analyze content and propose actions.
    This runs automatically — no human approval needed yet.
    """
    print(f"\n[Node: analyze_document] Analyzing content...")

    result = await analyze_chain.ainvoke({"content": state["content"]})

    # parse JSON response
    try:
        parsed = json.loads(result.strip())
        analysis = parsed.get("analysis", "")
        proposed_actions = parsed.get("proposed_actions", [])
    except json.JSONDecodeError:
        analysis = result
        proposed_actions = ["Review and process the document"]

    print(f"[Node: analyze_document] Proposed {len(proposed_actions)} actions")
    print(f"[Node: analyze_document] Waiting for human approval...")

    return {
        **state,
        "analysis": analysis,
        "proposed_actions": proposed_actions,
        "status": "awaiting_approval"
    }


# ── Node 2: Execute Actions ───────────────────────────────────────────────────

execute_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an action executor. Execute the approved actions
and report what was done. Be specific about each action taken."""),
    ("human", """Content: {content}

Approved actions to execute:
{actions}

Human feedback (if any): {feedback}

Report what you executed:""")
])

execute_chain = execute_prompt | llm | parser


async def execute_actions(state: HITLState) -> HITLState:
    """
    Node 2: Execute approved actions.
    Only runs after human approves.
    """
    print(f"\n[Node: execute_actions] Executing approved actions...")

    actions_text = "\n".join([
        f"{i+1}. {action}"
        for i, action in enumerate(state["proposed_actions"])
    ])

    result = await execute_chain.ainvoke({
        "content": state["content"],
        "actions": actions_text,
        "feedback": state.get("human_feedback", "No additional feedback")
    })

    print(f"[Node: execute_actions] Execution complete.")

    return {
        **state,
        "execution_result": result,
        "status": "completed"
    }


# ── Node 3: Handle Rejection ──────────────────────────────────────────────────

async def handle_rejection(state: HITLState) -> HITLState:
    """
    Node 3: Handle human rejection.
    Log the rejection and provide a clean response.
    """
    print(f"\n[Node: handle_rejection] Actions rejected by human.")

    return {
        **state,
        "execution_result": f"Actions were rejected by human reviewer. Feedback: {state.get('human_feedback', 'No feedback provided')}",
        "status": "rejected"
    }


# ── Conditional Edge ──────────────────────────────────────────────────────────

def route_after_review(state: HITLState) -> str:
    """
    Route based on human decision.
    approved → execute_actions
    rejected → handle_rejection
    """
    decision = state.get("human_decision", "").lower()
    if decision == "approved":
        return "execute_actions"
    else:
        return "handle_rejection"


# ── Build Graph ───────────────────────────────────────────────────────────────

# MemorySaver stores graph state between interrupts
# This is what allows the graph to pause and resume
checkpointer = MemorySaver()


def build_hitl_graph():
    """
    Build the human-in-the-loop graph.

    interrupt_before=["execute_actions"] means:
    - Graph runs analyze_document automatically
    - Graph PAUSES before execute_actions
    - Waits for human to update state with decision
    - Resumes with execute_actions or handle_rejection
    """
    graph = StateGraph(HITLState)

    graph.add_node("analyze_document", analyze_document)
    graph.add_node("execute_actions", execute_actions)
    graph.add_node("handle_rejection", handle_rejection)

    graph.set_entry_point("analyze_document")

    # conditional edge after analysis
    # routes based on human_decision in state
    graph.add_conditional_edges(
        "analyze_document",
        route_after_review,
        {
            "execute_actions": "execute_actions",
            "handle_rejection": "handle_rejection"
        }
    )

    graph.add_edge("execute_actions", END)
    graph.add_edge("handle_rejection", END)

    # compile with interrupt_before and checkpointer
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_actions", "handle_rejection"]
    )


hitl_graph = build_hitl_graph()

# store active pipeline runs
active_runs: dict = {}


async def start_pipeline(content: str) -> dict:
    """
    Start the HITL pipeline.
    Runs analyze_document then PAUSES waiting for human approval.
    Returns task_id and proposed actions for human to review.
    """
    task_id = str(uuid.uuid4())[:8]

    initial_state = HITLState(
        task_id=task_id,
        content=content,
        analysis="",
        proposed_actions=[],
        human_decision="",
        human_feedback="",
        execution_result="",
        status="started"
    )

    # config identifies this specific run in the checkpointer
    config = {"configurable": {"thread_id": task_id}}

    # store config for later resume
    active_runs[task_id] = config

    print(f"\n[HITL] Starting pipeline. Task ID: {task_id}")

    # run graph — will pause at interrupt point
    result = await hitl_graph.ainvoke(initial_state, config)

    return {
        "task_id": task_id,
        "analysis": result["analysis"],
        "proposed_actions": result["proposed_actions"],
        "status": result["status"],
        "message": "Pipeline paused. Review proposed actions and approve or reject."
    }


async def resume_pipeline(task_id: str,
                          decision: str,
                          feedback: str = "") -> dict:
    """
    Resume the pipeline after human review.
    Updates state with human decision then continues execution.
    """
    if task_id not in active_runs:
        return {"error": f"No active pipeline found for task_id: {task_id}"}

    config = active_runs[task_id]

    print(f"\n[HITL] Resuming pipeline {task_id} with decision: {decision}")

    # update state with human decision
    await hitl_graph.aupdate_state(
        config,
        {
            "human_decision": decision,
            "human_feedback": feedback
        }
    )

    # resume graph from where it paused
    result = await hitl_graph.ainvoke(None, config)

    # clean up completed run
    del active_runs[task_id]

    return {
        "task_id": task_id,
        "decision": decision,
        "execution_result": result["execution_result"],
        "status": result["status"]
    }


def get_pending_reviews() -> list:
    """
    Get all pipeline runs waiting for human review.
    """
    return [
        {"task_id": task_id, "status": "awaiting_approval"}
        for task_id in active_runs.keys()
    ]