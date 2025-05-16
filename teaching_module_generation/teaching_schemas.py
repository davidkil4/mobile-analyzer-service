from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator

# --- Input Schemas (from analyzer output JSON files) ---

class InputContextTurn(BaseModel):
    speaker: str
    text: str

class InputErrorDetail(BaseModel):
    type: str = Field(alias="type") # 'type' from JSON, can also be 'category'
    severity: str
    description: str = Field(alias="description") # 'description' from JSON, can also be 'error'
    correction: str

    class Config:
        populate_by_name = True # Allows using alias for field names

class InputRecommendation(BaseModel):
    """
    Represents a single "recommendation" item from the input JSON files.
    This is the primary data structure we'll pass to generation functions.
    """
    original: str
    corrected: str
    context: Optional[List[InputContextTurn]] = []
    focus_type: Optional[str] = None # e.g., "GRAMMAR", "PATTERNS", "VOCABULARY"
    reason: Optional[str] = None # Explanation for why this item was chosen
    selected_error: InputErrorDetail # The specific error to focus on

    utterance_index_in_file: Optional[int] = None 
    source_file_type: Optional[str] = None # "grammar", "patterns", "vocab"


# --- LLM Output Schemas (for Pydantic validation of LLM responses) ---

class LLMExplanation(BaseModel):
    introduction: str
    main: str
    recap: str

class LLMExplanationOutput(BaseModel):
    explanations: LLMExplanation

class LLMProblemFeedback(BaseModel):
    option_text: str
    is_correct: bool
    explanation: str

class LLMProblem(BaseModel):
    """
    Common structure for problems as expected from LLM output,
    before specific problem_id is assigned or dictation prompt_text is derived.
    """
    problem_id: Optional[str] = None # LLM might provide a placeholder or it's added later
    type: Literal["MCQ", "FillBlankChoice", "Dictation"]
    question: Optional[str] = None  # For MCQ, FillBlank (template)
    options: Optional[List[str]] = None # For MCQ, FillBlank
    feedback: Optional[List[LLMProblemFeedback]] = None # For MCQ, FillBlank
    target_text: Optional[str] = None # For Dictation (LLM provides this with brackets)
    prompt_text: Optional[str] = None # For Dictation (this will be derived from target_text)

    @validator('problem_id', pre=True, always=True)
    def set_problem_id_if_none(cls, v, values):
        return v

class LLMProblemsOutput(BaseModel):
    problems: List[LLMProblem]


# --- Final Teaching Module Schemas (for saving to output JSON) ---

class ProblemFeedback(BaseModel):
    option_text: str
    is_correct: bool
    explanation: str

class Problem(BaseModel):
    problem_id: str # Globally unique
    type: Literal["MCQ", "FillBlankChoice", "Dictation"]
    question: Optional[str] = None # For MCQ
    question_template: Optional[str] = None # For FillBlankChoice
    options: Optional[List[str]] = None
    feedback: Optional[List[ProblemFeedback]] = None
    prompt_text: Optional[str] = None # For Dictation (derived)
    target_text: Optional[str] = None # For Dictation (original from LLM, post-bracket processing if needed)


class TeachingModule(BaseModel):
    module_id: str
    module_type: str # e.g., "CONVERSATIONAL_LOW", "DRILL_PRACTICE"
    source_utterance_info: InputRecommendation
    explanations: LLMExplanation
    problems: List[Problem]
    help_context: Optional[str] = None

    class Config:
        # Support both Pydantic v1 and v2
        orm_mode = True  # For Pydantic v1 compatibility
        from_attributes = True  # For Pydantic v2 compatibility
