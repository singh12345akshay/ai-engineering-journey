"""
Seed script — populates ChromaDB with sample documents.
Run this after cloning the repo to get started immediately.

Usage: python scripts/seed_data.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embeddings import add_document

sample_docs = [
    {
        "doc_id": "rag_intro",
        "text": "If you want your AI app to answer questions from your own data, use RAG — Retrieval Augmented Generation. Instead of relying on the model memory, RAG retrieves relevant documents from your database and feeds them to the LLM as context. This way the model answers from your actual data not from its training.",
        "metadata": {"topic": "RAG", "source": "seed"}
    },
    {
        "doc_id": "fine_tuning",
        "text": "When you want to adapt a pre-trained model to perform well on your specific task or domain, use fine-tuning. You train the existing model on your own smaller dataset. This permanently changes the model weights to be better at your specific use case.",
        "metadata": {"topic": "fine-tuning", "source": "seed"}
    },
    {
        "doc_id": "vector_db",
        "text": "Vector databases help you search by meaning not by exact words. When a user searches for something, you convert their query to a vector and find the most similar vectors in the database. This is how semantic search works — finding relevant content even when the exact words dont match.",
        "metadata": {"topic": "vector-db", "source": "seed"}
    },
    {
        "doc_id": "langchain",
        "text": "LangChain helps you build AI applications faster. It connects LLMs with your data sources, manages conversation memory, and lets you chain multiple AI calls together. Use it when you want to build a chatbot, document QA system, or AI agent without writing everything from scratch.",
        "metadata": {"topic": "langchain", "source": "seed"}
    },
    {
        "doc_id": "prompt_engineering",
        "text": "Prompt engineering is how you get better answers from an LLM by designing better inputs. Instead of just asking a question, you give examples, tell the model to think step by step, or specify the exact format you want. Better prompts lead to dramatically better outputs without changing the model.",
        "metadata": {"topic": "prompt-engineering", "source": "seed"}
    }
]

if __name__ == "__main__":
    print("Seeding ChromaDB with sample documents...")
    for doc in sample_docs:
        add_document(doc["doc_id"], doc["text"], doc["metadata"])
        print(f"Added: {doc['doc_id']}")
    print(f"\nDone. Added {len(sample_docs)} documents.")
    print("Run: uvicorn app.main:app --reload")