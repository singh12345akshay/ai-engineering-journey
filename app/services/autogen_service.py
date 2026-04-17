"""
AutoGen multi-agent conversation service.
Agents have back-and-forth conversations to solve problems.
Uses autogen-agentchat 0.7.x API.
"""
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from app.config import settings


# ── Model Client (Groq via OpenAI-compatible endpoint) ───────────────────────

def _get_client() -> OpenAIChatCompletionClient:
    """Create an OpenAI-compatible client pointed at the Groq endpoint."""
    return OpenAIChatCompletionClient(
        model=settings.MODEL,
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_capabilities={
            "vision": False,
            "function_calling": True,
            "json_output": False,
        }
    )


async def run_autogen_debate(question: str) -> dict:
    """
    Run a 2-agent AutoGen conversation.
    AI Engineer answers, Devil's Advocate challenges,
    conversation continues until TERMINATE.
    """
    print(f"\n[AutoGen] Starting debate on: {question[:50]}...")

    client = _get_client()

    ai_engineer = AssistantAgent(
        name="AI_Engineer",
        model_client=client,
        system_message="""You are a senior AI engineer with deep expertise in
LLMs, RAG pipelines, and production AI systems.

When asked a question:
1. Provide a thorough technical answer
2. Include practical examples
3. Mention potential pitfalls
4. End your final answer with the word TERMINATE"""
    )

    devils_advocate = AssistantAgent(
        name="Devils_Advocate",
        model_client=client,
        system_message="""You are a critical thinker who challenges assumptions.

When the AI Engineer gives an answer:
1. Identify any weaknesses or missing considerations
2. Ask 1-2 pointed follow-up questions
3. Once satisfied with a complete answer, say "Good answer. TERMINATE"

Be constructive but rigorous."""
    )

    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        participants=[ai_engineer, devils_advocate],
        termination_condition=termination,
        max_turns=6
    )

    conversation = []
    final_answer = ""

    async for msg in team.run_stream(task=question):
        if hasattr(msg, "source") and hasattr(msg, "content"):
            conversation.append({
                "agent": msg.source,
                "message": str(msg.content)[:300]
            })
            if msg.source == "AI_Engineer":
                final_answer = str(msg.content).replace("TERMINATE", "").strip()

    print(f"[AutoGen] Debate complete. {len(conversation)} exchanges.")

    return {
        "question": question,
        "final_answer": final_answer,
        "conversation": conversation,
        "total_exchanges": len(conversation)
    }
