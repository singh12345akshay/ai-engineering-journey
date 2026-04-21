"""
Semantic Kernel demonstration.
Shows the plugin system and how SK differs from LangChain.
Run: python scripts/semantic_kernel_demo.py
"""
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory
from app.config import settings


# ── Step 1: Create the kernel ─────────────────────────────────────────────────

kernel = Kernel()

# add Groq as the LLM (OpenAI compatible)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import openai

# create custom openai client pointing to Groq
custom_client = openai.AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

service = OpenAIChatCompletion(
    ai_model_id=settings.MODEL,
    async_client=custom_client
)
kernel.add_service(service)

print("Kernel created with Groq LLM")


# ── Step 2: Define a Plugin ───────────────────────────────────────────────────

class InterviewPlugin:
    """
    A Semantic Kernel plugin for interview question management.
    Groups related functions into one cohesive unit.
    """

    @kernel_function(
        description="Categorise an interview question into domain and difficulty",
        name="categorise_question"
    )
    def categorise_question(self, question: str) -> str:
        """Categorise a question into domain and difficulty."""
        # In production this would call your zero-shot classifier
        # For demo we return a structured response
        return f"Question: {question}\nDomain: AI/ML\nDifficulty: Medium"

    @kernel_function(
        description="Generate a study guide for an interview topic",
        name="generate_study_guide"
    )
    def generate_study_guide(self, topic: str) -> str:
        """Generate a study guide outline for a topic."""
        return f"Study guide for {topic}: 1. Core concepts 2. Common questions 3. Practice problems"

    @kernel_function(
        description="Check if a question is a duplicate",
        name="check_duplicate"
    )
    def check_duplicate(self, question: str) -> str:
        """Check if a similar question already exists."""
        # In production this would search ChromaDB
        return f"No duplicate found for: {question[:50]}"


# register plugin with kernel
kernel.add_plugin(InterviewPlugin(), plugin_name="InterviewPlugin")
print("InterviewPlugin registered with kernel")


# ── Step 3: Semantic Function (prompt as a function) ─────────────────────────

# In Semantic Kernel, prompts are first-class functions
# You define them with template variables
ANSWER_SCORER_PROMPT = """
You are an expert interview coach. Score this interview answer.

Question: {{$question}}
Answer: {{$answer}}

Score on these dimensions (1-10 each):
1. Technical accuracy
2. Communication clarity
3. Completeness
4. Structure

Return as JSON:
{
  "technical": score,
  "communication": score,
  "completeness": score,
  "structure": score,
  "overall": average,
  "feedback": "one sentence of feedback"
}
"""

answer_scorer = kernel.add_function(
    function_name="score_answer",
    plugin_name="AnswerScorer",
    prompt=ANSWER_SCORER_PROMPT,
    prompt_execution_settings=OpenAIChatPromptExecutionSettings(
        max_tokens=300,
        temperature=0.1  # low temperature for consistent scoring
    )
)

print("AnswerScorer semantic function registered")


# ── Step 4: Memory and Chat History ──────────────────────────────────────────

async def run_interview_session():
    """
    Demonstrate Semantic Kernel chat with history.
    """
    print("\n" + "="*50)
    print("Semantic Kernel Interview Session Demo")
    print("="*50)

    history = ChatHistory()
    history.add_system_message(
        "You are an expert technical interviewer. "
        "Ask focused follow-up questions based on candidate answers."
    )

    # simulate an interview exchange
    exchanges = [
        "I would build a RAG pipeline by first chunking documents into 500 word pieces.",
        "I would use ChromaDB for vector storage and sentence-transformers for embeddings.",
        "For hybrid search I would combine BM25 with semantic search using an alpha parameter."
    ]

    history.add_user_message(
        "Tell me about how you would design a RAG pipeline."
    )

    for answer in exchanges:
        history.add_assistant_message(
            f"Interesting approach. {answer} Can you elaborate more on this?"
        )
        history.add_user_message(answer)

    # get final response
    chat_service = kernel.get_service(type=OpenAIChatCompletion)
    settings_obj = OpenAIChatPromptExecutionSettings(max_tokens=200)

    response = await chat_service.get_chat_message_content(
        chat_history=history,
        settings=settings_obj
    )

    print(f"\nFinal interviewer response:")
    print(response.content)


# ── Step 5: Compare LangChain vs Semantic Kernel ──────────────────────────────

def compare_frameworks():
    print("\n" + "="*50)
    print("LangChain vs Semantic Kernel Comparison")
    print("="*50)

    comparison = {
        "Language Support": {
            "LangChain": "Python only",
            "Semantic Kernel": "Python + C# + Java"
        },
        "Primary Use Case": {
            "LangChain": "Rapid prototyping, startups",
            "Semantic Kernel": "Enterprise, Microsoft stack"
        },
        "Abstraction Style": {
            "LangChain": "Chains and agents",
            "Semantic Kernel": "Plugins and functions"
        },
        "Memory": {
            "LangChain": "ConversationBufferMemory etc",
            "Semantic Kernel": "ChatHistory object"
        },
        "Azure Integration": {
            "LangChain": "Via langchain-azure package",
            "Semantic Kernel": "Native, first-class support"
        },
        "Best For You": {
            "LangChain": "Everything you've built so far",
            "Semantic Kernel": "If you join a Microsoft/Azure shop"
        }
    }

    for aspect, values in comparison.items():
        print(f"\n{aspect}:")
        print(f"  LangChain:        {values['LangChain']}")
        print(f"  Semantic Kernel:  {values['Semantic Kernel']}")


# ── Run everything ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compare_frameworks()
    asyncio.run(run_interview_session())