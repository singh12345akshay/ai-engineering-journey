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

## Key lesson from Day 5

Semantic search working correctly depends on two things equally:

1. Correct implementation (embeddings, ChromaDB, cosine similarity)
2. Document quality and query-document alignment

Most tutorials only teach point 1. Point 2 is what separates 
production RAG systems from tutorial demos.