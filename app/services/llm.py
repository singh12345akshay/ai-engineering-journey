import json
from openai import AsyncOpenAI
from app.config import settings
from app.models import UserQuery, AIResponse

client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Keep answers concise and clear."


async def call_llm(query: UserQuery) -> AIResponse:
    response = await client.chat.completions.create(
        model=settings.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query.question}
        ],
        max_tokens=query.max_tokens,
        temperature=query.temperature
    )

    return AIResponse(
        answer=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        model_used=settings.MODEL
    )


async def stream_tokens(query: UserQuery):
    try:
        stream = await client.chat.completions.create(
            model=settings.MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query.question}
            ],
            max_tokens=query.max_tokens,
            temperature=query.temperature,
            stream=True
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token is not None:
                # format each token as SSE
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

        # tell the client the stream is finished
        yield "data: [DONE]\n\n"

    except Exception as e:
        error = json.dumps({"error": str(e)})
        yield f"data: {error}\n\n"