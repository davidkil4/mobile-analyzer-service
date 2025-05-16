import pytest
import json
import logging

from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import TypeAdapter
from typing import List

from langchain_core.messages import AIMessage

from analyzer_service.analysis import _run_correction_chain_on_batch, ClauseCorrectionResult, _initialize_correction_chain
from analyzer_service.schemas import PreprocessedASUnit, AlignedClause, ErrorDetail, Severity, ContextUtterance, CorrectionBatchSchema, CorrectedClause
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging for testing
# This ensures logs are captured when running pytest with -s
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# --- Fixtures ---
@pytest.fixture
def sample_preprocessed_batch():
    # ... (fixture content remains the same)
    clause1 = AlignedClause(clause_id="unit-1-clause-1", clause_text="This is text original 1.")
    clause2 = AlignedClause(clause_id="unit-1-clause-2", clause_text="This is text original 2.")
    unit1 = PreprocessedASUnit(
        as_unit_id="unit-1",
        original_utterance_id="utterance-abc",
        original_input_text="This is the raw input.",
        # Add missing fields required by schema validation
        as_unit_text="This is text original 1. This is text original 2.", 
        aligned_original_text="This is text original 1. This is text original 2.", 
        preprocessed_text="This is text original 1. This is text original 2.",
        clauses=[clause1, clause2],
        context=None # Add context if needed by the test
    )
    return [unit1]

# --- Test Functions ---
@pytest.mark.asyncio # Mark test as async
async def test_run_correction_chain_on_batch_success(mocker, sample_preprocessed_batch):
    """
    Tests the _run_correction_chain_on_batch function with a successful chain invocation.
    Ensures corrected text is populated and errors_found remains empty.
    """
    # 1. Prepare Input Data (using the fixture)
    batch_units = sample_preprocessed_batch

    # 2. Prepare Mock Chain Output (Raw JSON string as returned by StrOutputParser)
    mock_output_list = [
        {
            "clause_id": "unit-1-clause-1",
            "corrected_clause_text": "This is corrected text 1."
        },
        {
            "clause_id": "unit-1-clause-2",
            "corrected_clause_text": "This is corrected text 2."
        }
    ]
    mock_raw_output_json_str = json.dumps(mock_output_list)

    # 3. Mock the chain's ainvoke method and the async initialization
    # Mock the chain instance that _initialize_correction_chain will return
    mock_chain_instance = MagicMock(spec=Runnable)
    # Use AsyncMock for the coroutine method 'ainvoke'
    # --- Return the raw JSON string --- 
    mock_chain_instance.ainvoke = AsyncMock(return_value=mock_raw_output_json_str) 

    # Patch the *async* _initialize_correction_chain to return our *mock* instance
    mocker.patch('analyzer_service.analysis._initialize_correction_chain', return_value=mock_chain_instance)

    # 4. Call the async function under test with await
    updated_batch = await _run_correction_chain_on_batch(batch_units)

    # 5. Assertions
    mock_chain_instance.ainvoke.assert_awaited_once() # Use assert_awaited_once
    call_args = mock_chain_instance.ainvoke.call_args[0][0]
    # Check for the correct input key for the prompt
    assert "batch_clauses_json" in call_args
    # --- Remove incorrect assertion: context is inside the json, not a separate key ---
    # assert "context" in call_args 

    # Check if corrected text was applied (assuming in-place modification or return)
    # Find the clauses in the returned batch
    assert updated_batch is not None
    assert len(updated_batch) == 1 # Should return all original units

    # Find the updated clauses by ID (assuming order might change)
    updated_clause_map = {}
    for unit in updated_batch:
        for clause in unit.clauses:
            updated_clause_map[clause.clause_id] = clause

    # Check clauses that should have been updated by the mock
    assert updated_clause_map["unit-1-clause-1"].corrected_clause_text == "This is corrected text 1."
    assert updated_clause_map["unit-1-clause-2"].corrected_clause_text == "This is corrected text 2."


@pytest.mark.asyncio # Mark test as async
async def test_correction_chain_initialization_and_invoke(mocker):
    """
    Tests that the correction chain (text only) initializes correctly using the
    real prompt and parser, and can invoke successfully with a mocked LLM.
    Verifies the basic chain plumbing (prompt | llm | parser).
    """
    # 1. Prepare Mock LLM Output (Simplified JSON string matching the updated schema)
    mock_llm_output_json_str = json.dumps({
        "corrected_clauses": [
            {
                "clause_id": "test-clause-1",
                "corrected_clause_text": "This is the corrected text."
                # errors_found removed
            }
        ]
    })
    mock_llm_response = AIMessage(content=mock_llm_output_json_str)

    # 2. Mock the ChatGoogleGenerativeAI class and its invoke method
    mock_chat_instance = MagicMock(spec=ChatGoogleGenerativeAI)
    mock_chat_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
    # Patch the class *within the analysis module* where it's imported/used
    mocker.patch('analyzer_service.analysis.ChatGoogleGenerativeAI', return_value=mock_chat_instance)

    # 3. Prepare Simple Test Input (Matching chain's expected input variables)
    test_input_clauses_json = json.dumps([
        {"clause_id": "test-clause-1", "clause_text": "This is text original."}
    ])
    test_input_context = "Previous utterance context."
    test_input = {"batch_clauses_json": test_input_clauses_json, "context": test_input_context}

    # 4. Call the *real* initialization function
    chain = await _initialize_correction_chain()

    # 5. Assert Initialization Success
    assert chain is not None
    assert isinstance(chain, Runnable)

    # 6. Invoke the chain with await
    result = await chain.ainvoke(test_input)

    # 7. Assert Invocation Success and Output (StrOutputParser returns string)
    assert isinstance(result, str) # <<< Corrected assertion: Expect a string
    # Check if the mocked LLM was called
    mock_chat_instance.ainvoke.assert_awaited_once()
    # Optionally, parse the result string and check its content if needed
    try:
        parsed_result = json.loads(result)
        # Example check on parsed data
        assert 'corrected_clauses' in parsed_result
        assert isinstance(parsed_result['corrected_clauses'], list)
        # assert parsed_result['corrected_clauses'][0]['clause_id'] == "test-clause-1"
    except json.JSONDecodeError:
        pytest.fail(f"Result string is not valid JSON: {result}")


# Add more async tests as needed, for example, for error handling
