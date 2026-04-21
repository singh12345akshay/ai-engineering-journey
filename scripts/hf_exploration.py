"""
HuggingFace exploration script.
Demonstrates the pipeline API and tokenizer internals.
Run: python scripts/hf_exploration.py
"""
from transformers import pipeline
import json


# ── Task 1: Text Classification ───────────────────────────────────────────────

print("\n" + "="*50)
print("Task 1: Sentiment Analysis")
print("="*50)

# pipeline automatically downloads the model on first run
# model: distilbert-base-uncased-finetuned-sst-2-english
# size: ~67MB — small and fast
sentiment = pipeline("sentiment-analysis")

texts = [
    "I love building AI systems from scratch",
    "Rate limits are frustrating when learning",
    "LangGraph makes agent flows predictable"
]

for text in texts:
    result = sentiment(text)
    print(f"Text: {text[:50]}")
    print(f"Sentiment: {result[0]['label']} ({result[0]['score']:.3f})")
    print()


# ── Task 2: Text Generation ───────────────────────────────────────────────────

print("="*50)
print("Task 2: Text Generation (local model)")
print("="*50)

# GPT-2 is small enough to run on any machine — 500MB
# Not as smart as Llama 3.3 70B but runs 100% locally
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=50
)

prompt = "A RAG pipeline works by"
result = generator(prompt, num_return_sequences=1)
print(f"Prompt: {prompt}")
print(f"Generated: {result[0]['generated_text']}")
print()


# ── Task 3: Question Answering ────────────────────────────────────────────────

print("="*50)
print("Task 3: Extractive Question Answering")
print("="*50)

# This model extracts answers directly from context
# Different from RAG — no retrieval, just extraction
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = """
Casual Leave is granted for 5 days per calendar year.
It cannot be carried forward to the next year.
CL is restricted to a maximum of 2 consecutive days.
Privilege Leave requires 1 week advance notice.
"""

questions = [
    "How many casual leaves are granted per year?",
    "Can casual leave be carried forward?",
    "What is the maximum consecutive CL days?"
]

for question in questions:
    result = qa(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (confidence: {result['score']:.3f})")
    print()


# ── Task 4: Zero-Shot Classification ─────────────────────────────────────────

print("="*50)
print("Task 4: Zero-Shot Classification")
print("="*50)

# Classify text into categories WITHOUT training on those categories
# This is powerful — works on any labels you define
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

interview_questions = [
    "Design a URL shortener that handles 1 billion requests per day",
    "What is your greatest weakness?",
    "Implement a binary search algorithm",
    "How would you build a RAG pipeline at scale?"
]

labels = ["System Design", "Behavioral", "DSA", "AI/ML"]

for question in interview_questions:
    result = classifier(question, candidate_labels=labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    print(f"Q: {question[:60]}")
    print(f"Category: {top_label} ({top_score:.3f})")
    print()
    
    
# ── Task 5: Tokenizer Internals ───────────────────────────────────────────────

print("="*50)
print("Task 5: Tokenizer Internals")
print("="*50)

from transformers import AutoTokenizer

# load the tokenizer for a model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

texts_to_tokenize = [
    "Hello world",
    "RAG pipeline",
    "unhappiness",
    "LangChain is a framework",
    "The quick brown fox"
]

for text in texts_to_tokenize:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"Text:   '{text}'")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(tokens)} tokens")
    print()

# show the vocabulary size
print(f"GPT-2 vocabulary size: {tokenizer.vocab_size} tokens")
print(f"That means {tokenizer.vocab_size} possible next tokens at each step")