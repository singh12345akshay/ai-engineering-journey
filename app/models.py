from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class UserQuery(BaseModel):
    question: str = Field(..., min_length=1)


class AIResponse(BaseModel):
    answer: str


class APIResponse(BaseModel):
    success: bool
    data: AIResponse


class PromptType(str, Enum):
    zero_shot = "zero_shot"
    few_shot = "few_shot"
    chain_of_thought = "chain_of_thought"
    structured_output = "structured_output"


class AdvancedQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    prompt_type: PromptType = Field(default=PromptType.zero_shot)
    max_tokens: int = Field(default=500, ge=50, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class AdvancedResponse(BaseModel):
    question: str
    prompt_type: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
class DocumentInput(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=10, max_length=5000)
    metadata: dict = Field(default={})


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    n_results: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    id: str
    text: str
    similarity: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    
class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str
    message: str
    
class LangChainQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default="default", min_length=1, max_length=100)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class LangChainResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None
    chain: str


class ConversationHistory(BaseModel):
    session_id: str
    history: list
    
class RAGQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    n_results: int = Field(default=3, ge=1, le=10)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    session_id: Optional[str] = None
    chain: str
    
class AdvancedRAGQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default="default")
    n_candidates: int = Field(default=10, ge=3, le=20)
    final_results: int = Field(default=3, ge=1, le=5)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self
    
class DocumentQARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    n_candidates: int = Field(default=10, ge=3, le=20)
    final_results: int = Field(default=3, ge=1, le=5)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    use_advanced: bool = Field(default=True)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class DocumentQAResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    session_id: str
    retrieval_method: str
    has_relevant_docs: bool
    
class AgentQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class AgentResponse(BaseModel):
    question: str
    answer: str
    status: str
    
class GraphQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class GraphResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    sources: list[str]
    chunks_used: int
    
class ResearchQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

    @model_validator(mode='after')
    def clean_question(self):
        self.question = self.question.strip()
        return self


class ResearchResponse(BaseModel):
    question: str
    answer: str
    quality_score: int
    sources: list[str]
    draft_answer: str
    critique: str
    
class HITLStartRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=2000)


class HITLResumeRequest(BaseModel):
    task_id: str = Field(..., min_length=1)
    decision: str = Field(..., pattern="^(approved|rejected)$")
    feedback: str = Field(default="")


class HITLStartResponse(BaseModel):
    task_id: str
    analysis: str
    proposed_actions: list[str]
    status: str
    message: str


class HITLResumeResponse(BaseModel):
    task_id: str
    decision: str
    execution_result: str
    status: str