import os
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings
from app.services.advanced_rag import advanced_rag_search
from app.services.embeddings import get_langchain_retriever
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# initialise the LLM once
# temperature=0.7 for general chat

llm = ChatOllama(
    model="llama3.2",
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


def get_memory(session_id: str):
    """Get or create message history for a session."""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ChatMessageHistory()
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
    history = memory.messages

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
    formatted = []

    for msg in memory.messages:
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
    history = memory.messages

    # build chain
    chain = conversational_rag_prompt | llm | parser

    # generate answer
    answer = await chain.ainvoke({
        "context": context,
        "question": question,
        "history": history
    })

    # save to memory
    memory.add_user_message(question)
    memory.add_ai_message(answer)

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
async def document_qa(request_data: dict) -> dict:
    """
    Production document Q&A endpoint.
    Combines everything:
    - Advanced RAG (hybrid search + reranking) or basic RAG
    - Conversation memory
    - Hallucination prevention
    - Source citation
    - LangSmith tracing (automatic)
    """
    question = request_data["question"]
    session_id = request_data["session_id"]
    use_advanced = request_data.get("use_advanced", True)
    n_candidates = request_data.get("n_candidates", settings.DEFAULT_N_CANDIDATES)
    final_results = request_data.get("final_results", settings.DEFAULT_FINAL_RESULTS)
    alpha = request_data.get("alpha", settings.DEFAULT_ALPHA)

    # retrieve relevant documents
    if use_advanced:
        docs = await advanced_rag_search(
            query=question,
            n_candidates=n_candidates,
            final_results=final_results,
            alpha=alpha
        )
    else:
        basic_docs = search_documents(
            query=question,
            n_results=final_results
        )
        docs = [
            {
                "text": d["text"],
                "metadata": d["metadata"],
                "hybrid_score": d["similarity"]
            }
            for d in basic_docs
        ]

    has_relevant_docs = len(docs) > 0

    if not has_relevant_docs:
        return {
            "answer": "I don't have any relevant documents to answer this question. Please upload relevant documents first.",
            "sources": [],
            "chunks_used": 0,
            "session_id": session_id,
            "retrieval_method": "advanced" if use_advanced else "basic",
            "has_relevant_docs": False
        }

    # format context with source citations
    context_parts = []
    for doc in docs:
        source = doc["metadata"].get("source", "unknown")
        page = doc["metadata"].get("page_or_slide", "?")
        score = doc.get("hybrid_score") or doc.get("rerank_score", 0)
        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # get conversation memory
    memory = get_memory(session_id)
    history = memory.load_memory_variables({})["history"]

    # generate answer
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

    # extract unique sources
    sources = list(set([
        f"{d['metadata'].get('source', 'unknown')} — page {d['metadata'].get('page_or_slide', '?')}"
        for d in docs
    ]))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(docs),
        "session_id": session_id,
        "retrieval_method": "hybrid_search + reranking" if use_advanced else "semantic_search",
        "has_relevant_docs": True
    }
    
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline


def get_local_llm():
    """
    Load a local HuggingFace model as a LangChain LLM.
    Uses GPT-2 for demonstration — replace with any model.
    In production use: mistralai/Mistral-7B-Instruct-v0.2
    """
    # create HuggingFace pipeline
    hf_pipe = hf_pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    # wrap as LangChain LLM
    local_llm = HuggingFacePipeline(pipeline=hf_pipe)

    return local_llm