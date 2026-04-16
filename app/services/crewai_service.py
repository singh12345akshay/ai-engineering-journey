"""
CrewAI service — role-based agent teams.
Each agent has a role, goal, and backstory.
Crew works together toward a shared objective.
"""
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from app.config import settings


# ── LLM ──────────────────────────────────────────────────────────────────────

# CrewAI LLM string format with rate limiting
llm = f"groq/{settings.MODEL}"

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate and comprehensive information",
    backstory="""You are a senior research analyst with expertise in AI.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2  # limit iterations to reduce API calls
)

writer = Agent(
    role="Technical Writer",
    goal="Transform research into clear content",
    backstory="""You are a technical writer specializing in AI concepts.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)

reviewer = Agent(
    role="Quality Assurance Editor",
    goal="Ensure content meets high quality standards",
    backstory="""You are a meticulous editor with AI expertise.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)


async def run_crew_research(question: str) -> dict:
    """
    Run a CrewAI research pipeline.
    3 agents with defined roles work together toward a shared goal.
    """
    print(f"\n[CrewAI] Starting crew research on: {question[:50]}...")

    # define tasks for each agent
    research_task = Task(
        description=f"""Research the following question thoroughly:
        {question}

        Find all relevant facts, examples, and considerations.
        Structure your findings as clear bullet points.
        Include specific details, numbers, and examples where relevant.""",
        agent=researcher,
        expected_output="Comprehensive research notes with key facts and examples"
    )

    writing_task = Task(
        description=f"""Using the research notes provided, write a clear and
        comprehensive answer to: {question}

        Requirements:
        - Start with a direct answer to the question
        - Follow with detailed explanation
        - Include practical examples
        - Use clear headings if the answer is long
        - Keep it concise but complete""",
        agent=writer,
        expected_output="Well-structured answer ready for review"
    )

    review_task = Task(
        description=f"""Review the written answer to: {question}

        Check for:
        1. Accuracy — are all facts correct?
        2. Completeness — does it fully answer the question?
        3. Clarity — is it easy to understand?
        4. Improvements — what could be better?

        Provide the final improved version of the answer.""",
        agent=reviewer,
        expected_output="Final reviewed and improved answer"
    )

    # create crew with sequential process
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,  # tasks run in order
        verbose=True
        max_rpm=10
    )

    # run the crew
    result = crew.kickoff()

    print(f"[CrewAI] Crew research complete.")

    return {
        "question": question,
        "answer": str(result),
        "agents_used": ["Senior Research Analyst", "Technical Writer", "QA Editor"],
        "process": "sequential"
    }