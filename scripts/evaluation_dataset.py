"""
Ground truth test dataset for RAG evaluation.
These are questions with known correct answers from your leave policy PDF.
The more questions you add, the more reliable your evaluation scores.
"""

EVALUATION_DATASET = [
    {
        "question": "How many casual leaves can an employee take at once?",
        "ground_truth": "Casual Leave is restricted to a maximum of two days in a stretch."
    },
    {
        "question": "How much notice is required for privilege leave?",
        "ground_truth": "Privilege Leave must be applied for at least 1 week in advance."
    },
    {
        "question": "How many days of bereavement leave can an employee take?",
        "ground_truth": "Employees can avail a maximum of 3 days of Bereavement Leave."
    },
    {
        "question": "Can casual leave be carried forward to next year?",
        "ground_truth": "Casual Leave cannot be carried forward to the next year and will lapse if not availed by end of year."
    }
]
    