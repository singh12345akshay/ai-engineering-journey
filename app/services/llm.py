import json
from openai import AsyncOpenAI
from app.config import settings
from app.models import UserQuery, AIResponse
from app.models import UserQuery, AIResponse, AdvancedQuery, AdvancedResponse

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
        
def build_prompt(question: str, prompt_type: str) -> list[dict]:
    if prompt_type == "zero_shot":
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer clearly and concisely."
            },
            {
                "role": "user",
                "content": question
            }
        ]

    elif prompt_type == "few_shot":
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": f"""Here are examples of how I want questions answered:

Q: What is Docker?
A: Docker is a containerization tool that packages your app and its
   dependencies into an isolated container — like a shipping container
   for software. Same contents run the same everywhere.
   Key benefit: eliminates the 'works on my machine' problem.

Q: What is an API?
A: An API is a contract between two systems defining how they communicate.
   Like a waiter in a restaurant — you tell the waiter what you want,
   they bring it from the kitchen. You never go to the kitchen yourself.
   Key benefit: systems communicate without knowing each other's internals.

Now answer in the exact same style — simple explanation, real world analogy, key benefit:
Q: {question}"""
            }
        ]

    elif prompt_type == "chain_of_thought":
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Always think step by step."
            },
            {
                "role": "user",
                "content": f"""Answer this question by thinking through it step by step.

Question: {question}

Think through it in this order:
Step 1: What problem does this solve?
Step 2: What are its core components?
Step 3: How do the components work together?
Step 4: When should someone use this?
Step 5: Give a one-line summary.

Show your thinking for each step, then give a final answer."""
            }
        ]

    elif prompt_type == "structured_output":
        return [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Always respond with valid JSON only. No extra text, no markdown, no code blocks."
            },
            {
                "role": "user",
                "content": f"""Answer this question as a JSON object with exactly these fields:

Question: {question}

Return this exact JSON structure:
{{
  "simple_definition": "one sentence explanation for a complete beginner",
  "technical_definition": "one sentence explanation for a software engineer",
  "key_components": ["component 1", "component 2", "component 3"],
  "when_to_use": "one sentence describing the ideal use case",
  "analogy": "explain using a real world analogy in one sentence"
}}

Return only the JSON. Nothing else."""
            }
        ]


async def ask_advanced(query: AdvancedQuery) -> AdvancedResponse:
    messages = build_prompt(query.question, query.prompt_type)

    response = await client.chat.completions.create(
        model=settings.MODEL,
        messages=messages,
        max_tokens=query.max_tokens,
        temperature=query.temperature
    )

    return AdvancedResponse(
        question=query.question,
        prompt_type=query.prompt_type,
        answer=response.choices[0].message.content,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )   