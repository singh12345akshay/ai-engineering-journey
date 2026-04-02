import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from models import UserQuery, AIResponse

load_dotenv()

# client = AsyncOpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# MODEL = "gemini-2.0-flash"

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.3-70b-versatile"

async def ask_ai(query: UserQuery) -> AIResponse:
    print(f"Sending question: '{query.question}'")
    print(f"Settings — max_tokens: {query.max_tokens}, temperature: {query.temperature}")

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Keep answers concise and clear."
            },
            {
                "role": "user",
                "content": query.question
            }
        ],
        max_tokens=query.max_tokens,
        temperature=query.temperature
    )

    return AIResponse(
        answer=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        model_used=MODEL
    )


async def main():

    print("\n=== TEST 1: Valid question ===")
    try:
        query = UserQuery(question="What is a RAG pipeline in AI?")
        result = await ask_ai(query)
        result.display()
    except ValidationError as e:
        print(f"Validation failed: {e}")

    print("\n=== TEST 2: Invalid input — too short ===")
    try:
        bad_query = UserQuery(question="Hi")
        result = await ask_ai(bad_query)
    except ValidationError as e:
        print(f"Pydantic blocked this before hitting the API: {e}")

    print("\n=== TEST 3: Custom settings ===")
    try:
        query2 = UserQuery(
            question="Explain async await in Python in one sentence",
            max_tokens=100,
            temperature=0.3
        )
        result2 = await ask_ai(query2)
        result2.display()
    except ValidationError as e:
        print(f"Validation failed: {e}")

    print("\n=== TEST 4: Empty string after stripping whitespace ===")
    try:
        query3 = UserQuery(question="   ")
    except ValidationError as e:
        print(f"Pydantic blocked whitespace-only input: {e}")


if __name__ == "__main__":
    asyncio.run(main())