# AI Engineer Journey

Documenting my transition from Frontend Developer (3 years) to AI Engineer.
Building in public — one project per week, every concept from scratch.

---

## The Goal

Land an AI Engineer role at 25 LPA within 4 months by building a strong
portfolio of production-grade AI projects.

---

## Roadmap

| Status | Phase | Timeline |
|--------|-------|----------|
| ✅ | Week 1 — Python for AI + LLM Foundations | Week 1 |
| ✅ | Week 2 — RAG Pipelines + LangChain | Week 2 |
| 🔄 | Week 3 — AI Agents + LangGraph | Week 3 |
| ⏳ | Week 4 — LangChain Deep Dive + HuggingFace | Week 4 |
| ⏳ | Month 2 — Fine-tuning + Cloud + MLOps | Month 2 |
| ⏳ | Month 3 — Advanced Projects + Interview Prep | Month 3 |
| ⏳ | Month 4 — Job Hunt | Month 4 |

---

## What I'm building

### Week 1 — Python for AI + LLM Foundations ✅

| Status | Day | Topic | What I built |
|--------|-----|-------|--------------|
| ✅ | Day 1 | Pydantic + Async LLM calls | AI Query Validator — typed, validated, async Groq API call with token tracking |
| ✅ | Day 2 | FastAPI + SSE Streaming | Production AI API — /ask (JSON) + /ask/stream (SSE) + /health endpoints |
| ✅ | Day 3 | Prompt Engineering | 4 prompting patterns — zero-shot, few-shot, CoT, structured output with token cost analysis |
| ✅ | Day 4 | How LLMs work | LLM internals — tokens, attention, RLHF, hallucinations, fine-tuning vs prompting |
| ✅ | Day 5 | Embeddings + Semantic Search | ChromaDB vector store, cosine similarity, query-document alignment lesson |
| ✅ | Day 6 | Document Parsing | PDF + PPTX + XLSX upload, chunking with overlap, seed script |
| ✅ | Day 7 | Week 1 Cleanup | Error handling, docstrings, README, Postman collection, LinkedIn post |

---

### Week 2 — RAG Pipelines + LangChain ✅

| Status | Day | Topic | What I built |
|--------|-----|-------|--------------|
| ✅ | Day 8 | LangChain Foundations | LCEL chains, ConversationBufferMemory, session isolation, LangSmith tracing |
| ✅ | Day 9 | Retrievers + Conversational RAG | ChromaDB as LangChain retriever, basic RAG + conversational RAG with memory |
| ✅ | Day 10 | Advanced RAG | Hybrid search (BM25 + semantic), cross-encoder reranking, alpha parameter |
| ✅ | Day 11 | RAGAS Evaluation | Faithfulness, answer relevancy, context recall, context precision metrics |
| ✅ | Day 12 | Production RAG API | /lc/document/qa endpoint, system status, multi-provider LLM fallback |
| ✅ | Day 13 | Week 2 Cleanup | Error handling, docstrings, README, LinkedIn post |
| ✅ | Day 14 | Week 2 Wrap | Final push, LinkedIn post, learnings doc update |

---

### Week 3 — AI Agents + LangGraph 🔄

| Status | Day | Topic | What I'll build |
|--------|-----|-------|-----------------|
| ✅ | Day 15 | Agent Foundations | ReAct pattern, tool calling, simple agent with web search + calculator tools |
| ✅ | Day 16 | LangGraph Basics | State machines, nodes, edges, conditional routing |
| ✅ | Day 17 | Multi-Agent Systems | Orchestrator + researcher + summarizer + critic agent |
| ✅ | Day 18 | Human in the Loop | Approval checkpoints, interrupt and resume |
| ⏳ | Day 19 | AutoGen + CrewAI | Role-based agent teams, multi-agent conversations |
| ⏳ | Day 20 | Agent API | LangGraph agent exposed as FastAPI endpoints with SSE streaming |
| ⏳ | Day 21 | Week 3 Cleanup | Clean up, LinkedIn post, README update |

---

### Week 4 — LangChain Deep Dive + HuggingFace ⏳

| Status | Day | Topic | What I'll build |
|--------|-----|-------|-----------------|
| ⏳ | Day 22 | LangChain Advanced | LCEL advanced, custom retrievers, callbacks, error handling |
| ⏳ | Day 23 | HuggingFace Basics | HF pipeline API, tokenizers, run Mistral locally with Ollama |
| ⏳ | Day 24 | Fine-Tuning Concepts | LoRA fine-tune on Colab free GPU, compare base vs fine-tuned |
| ⏳ | Day 25 | Semantic Kernel | Microsoft Semantic Kernel, plugin system |
| ⏳ | Day 26 | NLP Tools | BERT embeddings, SpaCy NER, integrate with RAG system |
| ⏳ | Day 27 | Week 4 Cleanup | LinkedIn post, README update |
| ⏳ | Day 28 | Month 1 Review | Full system test, fix gaps, prepare for Month 2 |

---

## Tech Stack

### Currently using
- Python 3.11
- FastAPI + Uvicorn
- Pydantic v2
- Groq API (Llama 3.3 70B) — primary LLM
- Gemini API (gemini-2.0-flash) — fallback
- Ollama (llama3.2) — local fallback when both exhausted
- LangChain + LCEL
- LangSmith — LLM monitoring and tracing
- ChromaDB — vector database
- sentence-transformers (all-MiniLM-L6-v2)
- rank-bm25 — hybrid search
- CrossEncoder (ms-marco-MiniLM-L-6-v2) — reranking
- pymupdf + python-pptx + openpyxl — document parsing
- RAGAS — RAG evaluation framework

### Coming in Month 2
- LangGraph — agent state machines
- AutoGen + CrewAI — multi-agent frameworks
- HuggingFace Transformers
- Docker + Kubernetes
- AWS / Azure
- MLflow

---

## Key Learnings

### Week 1
- SSE vs plain HTTP streaming — structured events vs raw bytes
- Prompt patterns — zero-shot (cheapest), few-shot (style), CoT (reasoning), structured (parseable)
- Query-document alignment — most common RAG failure mode
- LLM internals — tokens, attention (quadratic scaling), RLHF, hallucinations

### Week 2
- LangChain is structure not magic — LCEL pipes composable AI pipelines
- RAG = retrieval + generation — both layers must be tuned independently
- Hybrid search (alpha parameter) — keyword wins on abbreviations, semantic wins on concepts
- Re-ranking with cross-encoder improves precision after retrieval
- RAGAS measures what you can't see by eye — faithfulness, relevancy, recall, precision
- Rate limit management is a real production AI engineering skill

---

## How to run locally

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-engineer-journey.git
cd ai-engineer-journey

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env and add GROQ_API_KEY, GEMINI_API_KEY, LANGCHAIN_API_KEY

# Populate ChromaDB with sample documents
python scripts/seed_data.py

# Start the server
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` to see all endpoints.

---

## Daily commits

Every day this repo gets at least one commit. The green squares on my GitHub
profile are proof of consistency — not just claims of learning.
