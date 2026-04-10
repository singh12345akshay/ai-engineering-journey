import os
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings
from app.services.advanced_rag import advanced_rag_search
from app.services.embeddings import get_langchain_retriever

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

# ── Chain 4: RAG — Retrieval + Generation ───────────────────────────────────

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions
based ONLY on the provided context from documents.

Rules:
- If the answer is in the context, answer clearly and cite the source
- If the answer is NOT in the context, say exactly:
  "I don't have that information in the provided documents."
- Never make up information
- Always mention which document and page your answer comes from

Context:
{context}"""),
    ("human", "{question}")
])


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into a context string.
    Includes source and page number for citation.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_or_slide", "?")
        formatted.append(
            f"Source: {source} (page {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


async def ask_with_rag(question: str, n_results: int = 3) -> dict:
    """
    RAG chain — retrieves relevant chunks then generates answer.
    This is the complete Retrieval Augmented Generation pattern.
    """
    retriever = get_langchain_retriever(n_results)

    # retrieve relevant docs first
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I don't have any relevant documents to answer this question.",
            "sources": [],
            "chunks_used": 0,
            "chain": "rag"
        }

    # format docs into context string
    context = format_docs(docs)

    # build RAG chain
    rag_chain = rag_prompt | llm | parser

    # generate answer using retrieved context
    answer = await rag_chain.ainvoke({
        "context": context,
        "question": question
    })

    # extract sources for citation
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_or_slide", "?")
        sources.append(f"{source} — page {page}")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks_used": len(docs),
        "chain": "rag"
    }


# ── Chain 5: Conversational RAG — Memory + Retrieval ────────────────────────

conversational_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions
based on provided document context. You also remember the conversation history.

Rules:
- Answer using ONLY the provided context
- If information is not in context say "I don't have that information"
- Cite the source document and page number
- Use conversation history for follow-up questions

Context from documents:
{context}"""),
    ("placeholder", "{history}"),
    ("human", "{question}")
])


async def ask_conversational_rag(question: str,
                                  session_id: str,
                                  n_results: int = 3) -> dict:
    """
    Conversational RAG — combines memory + retrieval + generation.
    Remembers conversation history AND retrieves relevant documents.
    This is the full production RAG pattern.
    """
    retriever = get_langchain_retriever(n_results)
    memory = get_memory(session_id)

    # retrieve relevant docs
    docs = retriever.invoke(question)
    context = format_docs(docs) if docs else "No relevant documents found."

    # load conversation history
    history = memory.load_memory_variables({})["history"]

    # build chain
    chain = conversational_rag_prompt | llm | parser

    # generate answer
    answer = await chain.ainvoke({
        "context": context,
        "question": question,
        "history": history
    })

    # save to memory
    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    # extract sources
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_or_slide", "?")
        sources.append(f"{source} — page {page}")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks_used": len(docs),
        "session_id": session_id,
        "chain": "conversational_rag"
    }

async def ask_advanced_rag(question: str,
                            session_id: str,
                            n_candidates: int = 10,
                            final_results: int = 3) -> dict:
    """
    Advanced RAG — hybrid search + re-ranking + generation + memory.
    Production-grade retrieval pipeline.
    """
    # stage 1 + 2 — hybrid search + re-ranking
    reranked_docs = await advanced_rag_search(
        query=question,
        n_candidates=n_candidates,
        final_results=final_results
    )

    if not reranked_docs:
        return {
            "answer": "I don't have any relevant documents to answer this.",
            "sources": [],
            "chunks_used": 0,
            "retrieval_method": "advanced_rag",
            "session_id": session_id
        }

    # format context with scores for transparency
    context_parts = []
    for doc in reranked_docs:
        source = doc["metadata"].get("source", "unknown")
        page = doc["metadata"].get("page_or_slide", "?")
        context_parts.append(
            f"Source: {source} (page {page})\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # get conversation memory
    memory = get_memory(session_id)
    history = memory.load_memory_variables({})["history"]

    # generate answer using conversational RAG prompt
    chain = conversational_rag_prompt | llm | parser

    answer = await chain.ainvoke({
        "context": context,
        "question": question,
        "history": history
    })

    # save to memory
    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    # build sources
    sources = []
    for doc in reranked_docs:
        source = doc["metadata"].get("source", "unknown")
        page = doc["metadata"].get("page_or_slide", "?")
        sources.append(f"{source} — page {page}")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks_used": len(reranked_docs),
        "chain": "advanced_rag",
        "retrieval_method": "hybrid_search + reranking",
        "session_id": session_id
    }
