"""
Advanced LangChain patterns.
Covers: custom retrievers, callbacks, error handling, LCEL advanced.
Run: python scripts/advanced_langchain.py
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from app.config import settings
from typing import Any


llm = ChatGroq(
    model=settings.MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.7
)
parser = StrOutputParser()


# ── Pattern 1: RunnableParallel ───────────────────────────────────────────────
# Run multiple chains simultaneously and combine results

print("\n" + "="*50)
print("Pattern 1: RunnableParallel")
print("="*50)
print("Run multiple LLM calls at the same time")

async def demo_parallel():
    simple_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Give a simple one-sentence answer."),
        ("human", "{question}")
    ])

    technical_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Give a technical detailed answer for an engineer."),
        ("human", "{question}")
    ])

    # run both chains in parallel — saves time
    parallel_chain = RunnableParallel(
        simple=simple_answer_prompt | llm | parser,
        technical=technical_answer_prompt | llm | parser
    )

    result = await parallel_chain.ainvoke({
        "question": "What is RAG?"
    })

    print(f"\nSimple answer:\n{result['simple']}")
    print(f"\nTechnical answer:\n{result['technical'][:200]}...")


# ── Pattern 2: RunnableBranch ─────────────────────────────────────────────────
# Route to different chains based on input

print("\n" + "="*50)
print("Pattern 2: RunnableBranch")
print("="*50)
print("Route to different chains based on question type")

async def demo_branch():
    technical_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical expert. Give a detailed engineering answer."),
        ("human", "{question}")
    ])

    behavioral_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an interview coach. Give advice on answering behavioral questions."),
        ("human", "{question}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])

    # branch based on question content
    branch = RunnableBranch(
        (
            lambda x: any(word in x["question"].lower()
                         for word in ["design", "build", "implement", "code", "algorithm"]),
            technical_prompt | llm | parser
        ),
        (
            lambda x: any(word in x["question"].lower()
                         for word in ["weakness", "strength", "team", "conflict", "challenge"]),
            behavioral_prompt | llm | parser
        ),
        general_prompt | llm | parser  # default
    )

    questions = [
        "Design a URL shortener",
        "What is your greatest weakness?",
        "What is LangChain?"
    ]

    for question in questions:
        result = await branch.ainvoke({"question": question})
        print(f"\nQ: {question}")
        print(f"A: {result[:150]}...")


# ── Pattern 3: Custom Callbacks ───────────────────────────────────────────────
# Monitor and log every LLM call

print("\n" + "="*50)
print("Pattern 3: Custom Callbacks")
print("="*50)
print("Track token usage and latency on every call")

class TokenTracker(BaseCallbackHandler):
    """
    Custom callback that tracks token usage across all LLM calls.
    In production this feeds into your cost monitoring dashboard.
    """
    def __init__(self):
        self.total_tokens = 0
        self.total_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens = usage.get("total_tokens", 0)
            self.total_tokens += tokens
            self.total_calls += 1
            print(f"  [Callback] Call {self.total_calls}: {tokens} tokens used")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"  [Callback] LLM Error: {error}")


async def demo_callbacks():
    tracker = TokenTracker()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])

    chain = prompt | llm | parser

    questions = [
        "What is a vector database?",
        "What is the difference between RAG and fine-tuning?"
    ]

    for question in questions:
        await chain.ainvoke(
            {"question": question},
            config={"callbacks": [tracker]}
        )

    print(f"\nTotal calls: {tracker.total_calls}")
    print(f"Total tokens: {tracker.total_tokens}")


# ── Pattern 4: Error Handling in Chains ──────────────────────────────────────

print("\n" + "="*50)
print("Pattern 4: Error Handling")
print("="*50)
print("Graceful fallback when chain fails")

async def demo_error_handling():

    def validate_question(data: dict) -> dict:
        """Validate input before sending to LLM."""
        question = data.get("question", "")
        if len(question) < 5:
            raise ValueError(f"Question too short: '{question}'")
        if len(question) > 500:
            raise ValueError("Question too long")
        return data

    def fallback_response(error: Exception) -> str:
        """Return a safe fallback when the chain fails."""
        return f"Could not process this question. Error: {str(error)}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer concisely."),
        ("human", "{question}")
    ])

    # chain with validation and fallback
    chain = (
        RunnableLambda(validate_question)
        | prompt
        | llm
        | parser
    ).with_fallbacks([RunnableLambda(fallback_response)])

    test_cases = [
        {"question": "What is RAG?"},
        {"question": "Hi"},  # too short — will trigger fallback
    ]

    for test in test_cases:
        result = await chain.ainvoke(test)
        print(f"\nInput: {test['question']}")
        print(f"Result: {result[:150]}")


# ── Run all patterns ──────────────────────────────────────────────────────────

async def main():
    await demo_parallel()
    await demo_branch()
    await demo_callbacks()
    await demo_error_handling()

if __name__ == "__main__":
    asyncio.run(main())