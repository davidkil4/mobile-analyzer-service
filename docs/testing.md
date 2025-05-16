# Analyzer Service - Key Testing/Development Learnings & Pitfalls

This document summarizes critical challenges encountered and solutions implemented during the development and testing of the Analyzer Service chains (particularly Preprocessing, Clause Identification, and Batch Correction). Keep these points in mind when developing and testing new chains (like Accuracy Analysis) to avoid repeating past issues.

## 1. LLM Output Format & Parsing Issues

*   **Problem:** LLMs (especially for complex batch tasks like correction or segmentation) often failed to consistently produce the exact requested JSON format. Variations included malformed JSON, plain text, Markdown fences (` ```json `), unexpected nesting, and inconsistent dictionary keys.
*   **Impact:** Caused frequent `json.JSONDecodeError`, `pydantic.ValidationError`, `TypeError`, and `AttributeError` during parsing. Standard parsers (`JsonOutputParser`, Pydantic models) often failed.
*   **Solutions/Mitigation:**
    *   **Robust Prompts:** Explicitly define the desired output schema (using Pydantic models in the prompt), provide clear examples, and instruct the LLM *not* to use Markdown fences or conversational text. (Note: Still not foolproof).
    *   **Custom Parsers (Use Cautiously):** For the segmentation chain, where format was highly unreliable, we used `StrOutputParser` and wrote a robust Python parser (`preprocessing.parse_segmentation_output`) with multiple `try-except` blocks and fallback logic (e.g., line-by-line parsing) to handle various failure modes. *Avoid this complexity if possible*, as it makes code fragile.
    *   **Schema Enforcement:** Use Pydantic models rigorously to validate the structure *after* basic parsing succeeds.
    *   **Single, Focused Task:** Chains asking for *too much* structured data at once (e.g., correction + detailed errors + score) seem more prone to format failures. Consider separating complex tasks into distinct chains if instability arises.

## 2. Dependency Conflicts (Especially Pydantic v1 vs v2)

*   **Problem:** Incompatible versions of core dependencies, most notably `pydantic`, led to runtime errors like `TypeError: model_fields_schema() got an unexpected keyword argument 'ref_template'`. This often happened when different libraries (e.g., `langchain`, `fastapi`) required different major versions of Pydantic.
*   **Solution:**
    *   **Careful Version Pinning:** Use `pyproject.toml` (with Poetry) or `requirements.txt` to specify compatible versions. Check the dependency requirements of major libraries (like `langchain`). Aim for a consistent Pydantic version (preferably latest compatible v2) across all dependencies.
    *   **Clean Environment:** If conflicts persist, delete `poetry.lock` / `.venv` and run `poetry install` (or equivalent pip command) to force dependency resolution based on updated constraints.

## 3. Test Mocking & Assertion Failures (`AssertionError`)

*   **Problem:** Unit tests using `pytest` and `mocker` frequently failed with `AssertionError: Expected 'invoke'/'batch' to have been called...` or `...called with...`, indicating issues with mocking the LangChain chains.
*   **Solutions:**
    *   **Correct Mock Target Path:** This is critical. Use `mocker.patch` with the path to the chain object *where it is imported/used* in the module under test, not necessarily where it's defined. (e.g., `mocker.patch('analyzer_service.analysis.correction_chain')` if testing `analysis.py` which imports `correction_chain`).
    *   **Verify Code Path Execution:** Ensure test inputs/setup actually trigger the part of the code that calls the mocked chain method (`.invoke()` or `.batch()`). Check conditional logic.
    *   **Precise Assertions:** Use the correct assertion (`assert_called_once`, `assert_called_with`, `call_count`). When using `assert_called_with`, ensure the arguments match *exactly* what the code passes to the mock.

## 4. Input/Output Mismatches in Batch Processing

*   **Problem:** Occasionally, the number of items returned by an LLM in a batch response didn't match the number of items sent, causing errors during result alignment.
*   **Solution:** Always explicitly check `len(results) == len(inputs)` after a `chain.batch()` call. Log errors and implement a clear handling strategy (e.g., skip the batch, mark affected items, return partial results).

**General Recommendation for New Chains (Accuracy Analysis):**

*   **Start Simple:** Define the core task clearly.
*   **Prioritize Stability:** If requesting complex, multi-part structured output causes format instability (like we saw with correction), consider if the task can be broken down or simplified.
*   **Test Incrementally:** Write tests *early* using mocking. Ensure mocks are targeted correctly.
*   **Validate Dependencies:** Check for potential version conflicts before adding new libraries.