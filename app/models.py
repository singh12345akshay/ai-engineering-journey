from typing import Optional
from pydantic import BaseModel, Field, model_validator


class UserQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    max_tokens: int = Field(default=300, ge=50, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class AIResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_used: str

    def display(self):
        print("\n" + "─" * 40)
        print(f"Answer:\n{self.answer}")
        print("─" * 40)
        print(f"Prompt tokens:     {self.prompt_tokens}")
        print(f"Completion tokens: {self.completion_tokens}")
        print(f"Total tokens:      {self.total_tokens}")
        print(f"Model:             {self.model_used}")
        print("─" * 40 + "\n")


# NEW — this is what your /ask endpoint returns to the client
class APIResponse(BaseModel):
    success: bool
    data: Optional[AIResponse] = None
    error: Optional[str] = None