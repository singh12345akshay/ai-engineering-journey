# How LLMs Work — My Notes

## 1. Tokens
Tokens are the basic unit LLMs use to process text. A token can be a whole word, part of a word, or punctuation — for example 'unhappiness' might be split into 'un' and 'happiness'. The model never sees raw text — it converts everything into number IDs, then predicts the next number in the sequence. Both input and output are counted in tokens, which is how API usage is measured and billed. Rough rule: 1 token ≈ 0.75 words, so 1000 words ≈ 1,333 tokens

## 2. Context window
The context window is the model's temporary working memory — it holds everything the model can see at once: system prompt + past conversation + current input + model's output, all sharing the same token limit. Anything outside the window simply doesn't exist for the model — not partially forgotten, completely invisible. For Llama 3.3 70B on Groq this is 128,000 tokens. In RAG pipelines, retrieved documents are inserted into this window — so managing what goes in and how much space it takes is a critical production skill.

## 3. Temperature
Temperature controls the randomness of token selection. At temperature=0 the model always picks the most probable next token — same question asked 100 times gives the same answer every time. At temperature=0.7 (a good default) there's a balance between consistency and creativity. Near temperature=1 the model samples more broadly from possible tokens — more creative and varied but less predictable. Above 1 outputs can become incoherent. Practical rule: use low temperature (0.0-0.2) for structured output and factual tasks, higher temperature (0.7-1.0) for creative tasks.

## 4. Attention
Attention is the mechanism that calculates how relevant every token is to every other token in the context window — regardless of how far apart they are. When the model reads 'the trophy didn't fit because it was too big', attention figures out that 'it' refers to 'trophy' not 'suitcase' by assigning relevance scores between tokens. Important consequence: attention calculations are quadratic — double the context window = 4x the computation. This is why longer contexts are more expensive and why token optimization matters in production.

## 5. RLHF
RLHF happens in 3 stages. Stage 1 — Pre-training: the model trains on massive amounts of internet text and learns to predict the next token statistically. At this point it's knowledgeable but has no sense of being helpful or safe. Stage 2 — Supervised fine-tuning: human trainers write high quality example conversations — good questions with ideal answers. The model fine-tunes on these and starts behaving like an assistant. Stage 3 — Reinforcement learning from human feedback: the model generates multiple responses to the same question, human raters rank them by quality, and the model trains to maximise the reward score from those rankings. The ChatGPT thumbs up/thumbs down buttons are this process running continuously. RLHF is why models refuse harmful requests, say 'I don't know' instead of hallucinating, and have different personalities across providers.

## 6. Hallucinations
Hallucinations happen because the model predicts statistically likely tokens — not factually correct ones. It has no database of facts, no ability to verify information, and no sense of what it doesn't know. When asked something outside its training data it generates the most plausible sounding answer confidently — even if completely wrong. There are 3 main causes: knowledge gaps in training data, overconfident pattern matching on familiar question structures, and context loss in long conversations where early information falls outside the context window. RAG reduces hallucinations by grounding the model in retrieved factual documents instead of relying on training data recall — the model reads your actual data instead of guessing from memory.

## 7. Fine-tuning vs prompting
Prompting gives the model temporary instructions at inference time — it guides the output without changing anything permanent. Every request starts fresh. Fine-tuning permanently updates the model's weights by training on a specific dataset — changing the model itself. Use prompting when you need flexible, quickly changeable behaviour and when you're iterating fast. Use fine-tuning when you need consistent domain-specific behaviour across thousands of requests, have high quality labelled data, and have a clear performance gap that prompting alone can't close. For most startups — RAG and prompting first, fine-tune only when you have proven need and sufficient data.

---

## Q8. Why must you use the same embedding model for adding and searching?

Each embedding model creates vectors in its own mathematical space. If you 
add documents with model A and search with model B, the vectors live in 
incompatible spaces and similarity scores become meaningless.

Same model both ways — always.

---

## Q9. Why did the old documents return low similarity scores (0.1–0.2)?

The old documents were written as technical definitions using formal language. 
The query was conversational. The embedding model couldn't bridge that gap 
well enough to produce high similarity scores.

Old RAG document: "RAG stands for Retrieval Augmented Generation. 
It combines a retrieval system..."

New RAG document: "If you want your AI app to answer questions from 
your own data, use RAG..."

The new version directly mirrors the query language — producing scores 
of 0.5–0.7 instead of 0.1–0.2.

---

## Q10. What is query-document alignment and why does it matter?

Query-document alignment means writing documents in language that matches 
how users will actually query them. When document language and query language 
are similar, the embedding model produces high similarity scores and retrieval 
works correctly.

This is one of the most common production RAG failure modes — correct 
implementation, wrong results, because documents and queries are written 
in completely different styles.

In production RAG systems, document preprocessing and chunking strategy 
directly affects retrieval quality. This is why it is a core AI engineering 
skill.

---

## Q11. What would happen if you searched for something completely 
unrelated to all documents?

ChromaDB would still return the n_results closest documents — it always 
returns something. But the similarity scores would be very low (0.05–0.15). 

In production you handle this by setting a minimum similarity threshold 
and returning "no relevant results found" if nothing scores above it:
```python
results = [r for r in results if r["similarity"] > 0.4]
if not results:
    return {"message": "No relevant documents found"}
```

---

---

## Week 3 — AI Agents + LangGraph

### Q12. What is the ReAct pattern and why does it work?

ReAct = Reason + Act. Instead of answering directly, the agent produces a
Thought (what it needs to find), then an Action (which tool to call), then
observes the result, then reasons again. This loop continues until it has
enough information to give a final answer.

Why it works: the model can decompose a multi-step question into smaller
lookups, verify intermediate results, and course-correct if a tool returns
nothing useful. Without this loop, the model either guesses or hallucinates
for anything outside its training data.

---

### Q13. What is LangGraph and how is it different from a plain LangChain chain?

A LangChain chain is linear — input goes through steps A → B → C and exits.
LangGraph is a directed graph — steps are nodes, transitions are edges, and
the path through the graph depends on the current state. Nodes can loop,
branch, wait for human input, or call other subgraphs.

LangGraph adds three things chains cannot express:
1. Conditional branching — go to node X or node Y based on the state
2. Loops — retry, self-correct, or escalate
3. Persistent state — pause execution, store state, resume later

In production this matters because real agent tasks are not linear. A
research task might need 3 rounds of refinement. A document processing
task might need human approval. A chatbot might need to branch on intent.

---

### Q14. How does human-in-the-loop work in LangGraph?

HITL uses LangGraph's interrupt mechanism:
1. `start_pipeline` runs the graph until it hits an interrupt node, then
   saves the graph state and returns a `task_id`
2. A human reviews the proposed action via the API
3. `resume_pipeline` loads the saved state by `task_id`, injects the human's
   decision (approved/rejected + optional feedback), and continues the graph

The key engineering insight: HITL is a state persistence problem. The graph
state must be serialisable and storable so it can be resumed in a different
process or after a server restart. In production you'd use a database or
Redis instead of an in-memory dict.

---

### Q15. What is the difference between AutoGen and CrewAI?

**AutoGen** is conversation-based — agents send messages to each other in a
chat loop. The pattern is natural dialogue: one agent responds, another
replies, until a termination condition is met. Good for debate, critique,
and back-and-forth refinement.

**CrewAI** is task-based — agents have roles, goals, and backstories, and
are assigned tasks with expected outputs. The Crew runs tasks sequentially
(or in parallel) with each agent picking up where the last left off.
Good for pipelines where each step needs a different specialisation.

When to use which:
- Need multiple perspectives on the same question → AutoGen
- Need a pipeline of specialists (researcher → writer → reviewer) → CrewAI
- Need agents to autonomously hand off work with defined quality gates → CrewAI

---

### Q16. Why does SSE streaming matter for agent endpoints?

Without streaming, the client fires a request and waits silently for 30-90
seconds while 3-5 LLM calls run sequentially. This looks broken.

With SSE streaming, the server emits events as each agent step completes:
```
data: {"type": "status", "message": "Researcher is working..."}
data: {"type": "result", "step": "research", "content": "..."}
data: {"type": "status", "message": "Writer is working..."}
data: {"type": "done", "answer": "..."}
```

The frontend shows real-time progress, which makes a 60-second pipeline
feel responsive instead of broken.

---

## Key lesson from Day 5

Semantic search working correctly depends on two things equally:

1. Correct implementation (embeddings, ChromaDB, cosine similarity)
2. Document quality and query-document alignment

Most tutorials only teach point 1. Point 2 is what separates 
production RAG systems from tutorial demos.