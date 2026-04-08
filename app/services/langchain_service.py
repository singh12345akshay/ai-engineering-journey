import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings

# initialise the LLM once
# temperature=0.7 for general chat
llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.7
)

# output parser — extracts text string from LLM response
parser = StrOutputParser()

# ── Chain 1: Simple QA ──────────────────────────────────────────────────────
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Keep answers concise and clear."),
    ("human", "{question}")
])

simple_chain = simple_prompt | llm | parser

async def ask_simple(question: str) -> dict:
    """
    Simple QA chain — replaces your direct Groq API call.
    Same result, but now traceable in LangSmith.
    """
    answer = await simple_chain.ainvoke({"question": question})
    return {"answer": answer, "chain": "simple"}


# ── Chain 2: Prompted QA with system instruction ────────────────────────────
def build_prompted_chain(system_instruction: str):
    """
    Build a chain with a custom system instruction.
    Returns a new chain — not a fixed one.
    This is the factory pattern for chains.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{question}")
    ])
    return prompt | llm | parser


# ── Chain 3: Conversation with memory ───────────────────────────────────────

# store conversation histories per session
# in production you'd use Redis or a database
conversation_memories: dict = {}


def get_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create memory for a session."""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
    return conversation_memories[session_id]


conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the conversation history to provide contextual responses."),
    ("placeholder", "{history}"),
    ("human", "{question}")
])


async def ask_with_memory(question: str, session_id: str) -> dict:
    """
    Conversational chain — remembers previous messages.
    Same session_id = same conversation history.
    """
    memory = get_memory(session_id)

    # load existing history
    history = memory.load_memory_variables({})["history"]

    # build chain
    chain = conversation_prompt | llm | parser

    # get answer
    answer = await chain.ainvoke({
        "question": question,
        "history": history
    })

    # save this exchange to memory
    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    return {
        "answer": answer,
        "session_id": session_id,
        "chain": "conversation_with_memory"
    }


def clear_memory(session_id: str) -> dict:
    """Clear conversation history for a session."""
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "no session found", "session_id": session_id}


def get_conversation_history(session_id: str) -> dict:
    """Get conversation history for a session."""
    if session_id not in conversation_memories:
        return {"session_id": session_id, "history": []}

    memory = conversation_memories[session_id]
    history = memory.load_memory_variables({})["history"]

    formatted = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})

    return {"session_id": session_id, "history": formatted}