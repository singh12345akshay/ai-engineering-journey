"""
AI Agent service using ReAct pattern.
Agent reasons about which tools to use to answer a question.
"""
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from app.config import settings
from app.services.embeddings import search_documents


# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0
    # temperature=0 for agents — consistent tool selection
    # higher temperature makes agents unpredictable
)


# ── Tools ────────────────────────────────────────────────────────────────────

# Tool 1 — Web search
search_tool = DuckDuckGoSearchRun()

# Tool 2 — Wikipedia
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# Tool 3 — Calculator
@tool
def calculator(expression: str) -> str:
    """
    Useful for doing math calculations.
    Input should be a valid Python mathematical expression.
    Example: '2 + 2', '100 * 0.15', '(50 + 30) / 4'
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Tool 4 — Document search (your RAG system as a tool)
@tool
def document_search(query: str) -> str:
    """
    Useful for searching through uploaded company documents,
    policies, and internal knowledge base.
    Use this when the question is about internal company information,
    leave policies, or any uploaded documents.
    Input should be a clear search query.
    """
    results = search_documents(query=query, n_results=3)

    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        page = r["metadata"].get("page_or_slide", "?")
        formatted.append(f"[{source} p.{page}]: {r['text']}")

    return "\n\n".join(formatted)


# All tools available to the agent
tools = [search_tool, wiki_tool, calculator, document_search]


# ── ReAct Prompt ─────────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant with access to tools.
Use the tools to answer the user's question accurately.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


# ── Agent ────────────────────────────────────────────────────────────────────

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=REACT_PROMPT
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # prints reasoning steps to terminal
    max_iterations=5,       # stop after 5 tool uses max
    handle_parsing_errors=True  # handle malformed LLM outputs
)


async def run_agent(question: str) -> dict:
    """
    Run the ReAct agent on a question.
    Agent decides which tools to use and reasons through the answer.
    """
    try:
        result = await agent_executor.ainvoke({"input": question})

        return {
            "question": question,
            "answer": result["output"],
            "status": "success"
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"Agent error: {str(e)}",
            "status": "error"
        }