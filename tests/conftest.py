import pytest
from typing import List

# Add project root to sys.path if necessary, or ensure PYTHONPATH is set
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary schemas from the main application
# Handle potential ImportError during test collection if schemas aren't found
try:
    from analyzer_service.schemas import PreprocessedASUnit, AlignedClause, ContextUtterance
except ImportError:
    # Define basic placeholders if import fails during collection
    class BaseModel:
        pass
    class PreprocessedASUnit(BaseModel):
        def __init__(self, **kwargs): pass
    class AlignedClause(BaseModel):
         def __init__(self, **kwargs): pass
    class ContextUtterance(BaseModel):
         def __init__(self, **kwargs): pass

@pytest.fixture(scope="module")
def sample_preprocessed_batch() -> List[PreprocessedASUnit]:
    """Provides a sample list of PreprocessedASUnit objects for testing."""
    # Create sample data structures matching the schemas
    unit1 = PreprocessedASUnit(
        as_unit_id="unit-1",
        as_unit_text="This is the first unit.",
        original_utterance_id="utterance-abc",
        original_input_text="This is the raw first unit?",
        aligned_original_text="This is the first unit.",
        clauses=[
            AlignedClause(clause_id="unit-1-clause-1", clause_text="This is the first clause.", errors_found=[]),
            AlignedClause(clause_id="unit-1-clause-2", clause_text="This is the second clause.", errors_found=[]),
        ],
        context=[
            ContextUtterance(speaker="User", text="Previous user utterance."),
            ContextUtterance(speaker="AI", text="Previous AI response.")
        ]
        # Add other fields from PreprocessedASUnit if they exist and are needed
    )

    unit2 = PreprocessedASUnit(
        as_unit_id="unit-2",
        as_unit_text="This is unit number two.",
        original_utterance_id="utterance-def",
        original_input_text="Unit number two here.",
        aligned_original_text="This is unit number two.",
        clauses=[
            AlignedClause(clause_id="unit-2-clause-1", clause_text="Clause three overall.", errors_found=[]),
        ],
        context=None # Example with no context
    )

    return [unit1, unit2]
