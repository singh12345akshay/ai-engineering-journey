"""
NLP Tools demonstration — SpaCy + BERT.
Shows when to use specialised NLP models instead of LLMs.
Run: python scripts/nlp_tools_demo.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np


# ── Task 1: Named Entity Recognition with SpaCy ───────────────────────────────

print("\n" + "="*60)
print("Task 1: Named Entity Recognition (SpaCy)")
print("="*60)
print("Extract company names and tech terms from interview text")
print()

nlp = spacy.load("en_core_web_sm")

interview_texts = [
    "I interviewed at Google for a Senior Software Engineer role. The process had 5 rounds including system design and coding.",
    "Amazon asked me about designing a URL shortener during my SDE-2 interview in Bangalore.",
    "My Meta interview experience: they asked about RAG pipelines and LangChain in the AI Engineer round.",
    "Flipkart and Razorpay both asked similar questions about Redis caching and PostgreSQL optimisation."
]

for text in interview_texts:
    doc = nlp(text)

    # extract named entities
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    print(f"Text: {text[:70]}...")
    print(f"  Companies found: {companies}")
    print(f"  Locations found: {locations}")
    print()


# ── Task 2: Keyword Extraction ────────────────────────────────────────────────

print("="*60)
print("Task 2: Technical Keyword Extraction")
print("="*60)
print("Extract important technical terms using POS tagging")
print()

def extract_technical_keywords(text: str) -> list:
    """
    Extract technical keywords using SpaCy POS tagging.
    Nouns and proper nouns are usually the important terms.
    Filters out common stop words.
    """
    doc = nlp(text)
    keywords = []

    for token in doc:
        if (token.pos_ in ["NOUN", "PROPN"] and
            not token.is_stop and
            len(token.text) > 2 and
            token.text.lower() not in ["round", "interview", "question", "answer"]):
            keywords.append(token.text.lower())

    # deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


interview_questions = [
    "Design a distributed rate limiter that handles 1 billion requests per day",
    "How would you build a RAG pipeline with hybrid search and cross-encoder reranking?",
    "Implement a binary search tree with insert, delete, and search operations",
    "Tell me about a time you resolved a conflict with a team member"
]

for question in interview_questions:
    keywords = extract_technical_keywords(question)
    print(f"Q: {question[:60]}...")
    print(f"   Keywords: {keywords[:6]}")
    print()


# ── Task 3: BERT Embeddings for Similarity ────────────────────────────────────

print("="*60)
print("Task 3: Question Similarity with BERT Embeddings")
print("="*60)
print("Find similar questions without calling an LLM")
print()

# use sentence-transformers (wraps BERT for sentence embeddings)
# you already have this installed from Week 1
model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# database of existing questions
existing_questions = [
    "Design a URL shortener service",
    "How does attention mechanism work in transformers?",
    "What is the difference between process and thread?",
    "Design a notification system for 1 million users",
    "Explain ACID properties in databases",
    "How would you implement LRU cache?"
]

# new questions to check for duplicates
new_questions = [
    "Build a link shortening service like bit.ly",  # similar to URL shortener
    "Explain self-attention in neural networks",     # similar to attention mechanism
    "Design an alert system for large scale apps",   # similar to notification system
    "What is LRU cache and how to implement it?"    # duplicate
]

# embed all questions
existing_embeddings = model.encode(existing_questions)

print("Checking new questions for duplicates:\n")
for new_q in new_questions:
    new_embedding = model.encode(new_q)
    similarities = [
        cosine_similarity(new_embedding, existing_emb)
        for existing_emb in existing_embeddings
    ]
    max_similarity = max(similarities)
    most_similar_idx = similarities.index(max_similarity)
    most_similar_q = existing_questions[most_similar_idx]

    status = "DUPLICATE" if max_similarity > 0.85 else \
             "SIMILAR" if max_similarity > 0.65 else "NEW"

    print(f"New Q:    {new_q[:55]}")
    print(f"Similar:  {most_similar_q[:55]}")
    print(f"Score:    {max_similarity:.3f}  [{status}]")
    print()


# ── Task 4: Text Classification without LLM ──────────────────────────────────

print("="*60)
print("Task 4: Question Difficulty Classification")
print("="*60)
print("Classify difficulty using embeddings — no LLM call needed")
print()

# reference embeddings for each difficulty level
difficulty_descriptions = {
    "Easy": "basic simple fundamental beginner concept definition what is",
    "Medium": "implement design explain difference when how would you",
    "Hard": "scale distributed system design million billion optimize tradeoff"
}

difficulty_embeddings = {
    level: model.encode(desc)
    for level, desc in difficulty_descriptions.items()
}

test_questions = [
    "What is a linked list?",
    "Implement a binary search algorithm",
    "Design a distributed cache system handling 10 million requests per second",
    "What is recursion?",
    "How would you implement a rate limiter?",
    "Design Google's search indexing system"
]

for question in test_questions:
    q_embedding = model.encode(question)

    similarities = {
        level: cosine_similarity(q_embedding, emb)
        for level, emb in difficulty_embeddings.items()
    }

    predicted_difficulty = max(similarities, key=similarities.get)
    confidence = similarities[predicted_difficulty]

    print(f"Q: {question[:55]}")
    print(f"   Difficulty: {predicted_difficulty} (confidence: {confidence:.3f})")
    print()


# ── Task 5: SpaCy vs LLM comparison ──────────────────────────────────────────

print("="*60)
print("Task 5: Speed Comparison — SpaCy vs LLM")
print("="*60)

import time

test_text = "Google asked me about designing a RAG pipeline at Amazon scale during my Meta interview in Bangalore."

# SpaCy NER timing
start = time.time()
for _ in range(100):
    doc = nlp(test_text)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
spacy_time = (time.time() - start) / 100 * 1000

print(f"SpaCy NER (100 runs average):")
print(f"  Time:     {spacy_time:.2f}ms per call")
print(f"  Result:   {companies}")
print(f"  API cost: $0.00")
print()
print(f"LLM NER (estimated):")
print(f"  Time:     500-2000ms per call")
print(f"  Result:   same companies")
print(f"  API cost: ~$0.001 per call")
print()
print(f"Speed advantage: {500/spacy_time:.0f}x faster")
print(f"Cost advantage:  infinite (SpaCy is free)")