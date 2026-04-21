"""
Fine-tuning dataset for AI engineering Q&A.
Format: instruction → response pairs.
This teaches the model to answer AI engineering questions correctly.
"""

FINETUNE_DATA = [
    {
        "instruction": "What is a RAG pipeline?",
        "response": "RAG stands for Retrieval Augmented Generation. It combines a retrieval system with a language model to generate accurate answers grounded in real documents. Instead of relying on the model's training data, RAG retrieves relevant chunks from your document database and feeds them to the LLM as context."
    },
    {
        "instruction": "What is the difference between fine-tuning and RAG?",
        "response": "RAG retrieves external documents at inference time to ground the model's answers. Fine-tuning permanently updates the model's weights by training on a specific dataset. RAG is better for dynamic knowledge that changes frequently. Fine-tuning is better for consistent domain-specific behaviour and style."
    },
    {
        "instruction": "What is LangChain?",
        "response": "LangChain is a framework for building applications with large language models. It provides tools for chaining LLM calls, managing conversation memory, connecting to vector stores, and building agent systems. It abstracts the complexity of working with LLMs into composable building blocks."
    },
    {
        "instruction": "What is LangGraph?",
        "response": "LangGraph is a library built on top of LangChain for building stateful multi-agent systems using graph-based state machines. You define nodes (functions that process state), edges (connections between nodes), and conditional edges (routing based on state values). It gives you explicit control over agent flow instead of letting the LLM decide everything."
    },
    {
        "instruction": "What is ChromaDB?",
        "response": "ChromaDB is an open-source vector database that stores embeddings alongside metadata for semantic similarity search. It runs embedded inside your Python process with no separate server needed, similar to SQLite. It uses cosine similarity to find documents with similar meaning to a query."
    },
    {
        "instruction": "What is LoRA in the context of fine-tuning?",
        "response": "LoRA stands for Low-Rank Adaptation. Instead of updating all model weights during fine-tuning, LoRA adds two small matrices alongside each layer and only trains those. This reduces trainable parameters by 100-1000x while achieving similar quality to full fine-tuning. The original weights stay frozen, preventing catastrophic forgetting."
    },
    {
        "instruction": "What is cosine similarity?",
        "response": "Cosine similarity measures the angle between two vectors rather than their distance. A score of 1.0 means identical direction (same meaning), 0.5 means somewhat related, 0.0 means no relationship. It is used in semantic search to find documents whose embedding vectors point in a similar direction to the query embedding."
    },
    {
        "instruction": "What is RAGAS?",
        "response": "RAGAS is a framework for evaluating RAG pipelines using four metrics: faithfulness (is the answer supported by retrieved context?), answer relevancy (does the answer address the question?), context recall (did retrieval find the right chunks?), and context precision (were retrieved chunks actually relevant?). It makes RAG quality measurable instead of subjective."
    },
    {
        "instruction": "What is the ReAct pattern in AI agents?",
        "response": "ReAct stands for Reasoning and Acting. The agent follows a loop: Thought (reason about what to do), Action (call a tool), Observation (read the result), then loop back to Thought. This continues until the agent has enough information to produce a Final Answer. Temperature should be 0 for agents to ensure consistent tool selection."
    },
    {
        "instruction": "What is chunking in RAG?",
        "response": "Chunking splits documents into smaller pieces before embedding. Embedding a whole document averages all meaning into one vector, losing specificity. Chunks of 300-500 words each get their own vector, allowing precise retrieval of specific sections. Overlap between chunks (50-100 words) preserves context at boundaries."
    },
    {
        "instruction": "What is hybrid search?",
        "response": "Hybrid search combines BM25 keyword search with semantic vector search. BM25 finds exact word matches (good for abbreviations, codes, names). Semantic search finds meaning-based matches (good for paraphrases). The alpha parameter controls the balance: alpha=0 is pure keyword, alpha=1 is pure semantic, alpha=0.5 is balanced hybrid."
    },
    {
        "instruction": "What is human in the loop in AI systems?",
        "response": "Human in the loop means pausing an automated pipeline at critical decision points for human review before continuing. In LangGraph this is implemented with interrupt_before which pauses execution before a specified node, saves state with MemorySaver, and waits for human input. Essential for high-stakes actions like sending emails, deleting data, or financial transactions."
    },
]


def format_for_training(example: dict) -> str:
    """
    Format a Q&A pair as an instruction-response string.
    This is the standard format for instruction fine-tuning.
    """
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}

### End"""


if __name__ == "__main__":
    print(f"Dataset size: {len(FINETUNE_DATA)} examples")
    print("\nSample formatted example:")
    print(format_for_training(FINETUNE_DATA[0]))