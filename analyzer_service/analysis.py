import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import math
import asyncio
import csv

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

# Pydantic for schema validation
from pydantic import BaseModel, Field, validator, ValidationError

try:
    # Use relative import for schemas
    from .schemas import PreprocessedASUnit, MainAnalysisOutput, AlignedClause, ErrorDetail, Severity, ContextUtterance, CorrectionBatchSchema, AccuracyAnalysisResult, AccuracyBatchSchema, PatternDetail, PatternAnalysisResult, PatternAnalysisBatchSchema, ClauseAnalysis, FilteringMetadata
except ImportError:
    logging.warning("Schemas not found in analysis.py, using placeholders.")
    # Define placeholders if import fails (useful for isolated testing/dev)
    class BaseModel:
        pass # Basic placeholder
    class PreprocessedASUnit(BaseModel): pass
    class MainAnalysisOutput(BaseModel): pass
    class AlignedClause(BaseModel): 
        errors_found: List[ErrorDetail] = Field([], description="List of errors found. Empty list if none.")
    class ErrorDetail(BaseModel): pass
    class Severity(BaseModel): pass
    class ContextUtterance(BaseModel): pass
    class CorrectionBatchSchema(BaseModel): pass
    class AccuracyAnalysisResult(BaseModel): pass
    class AccuracyBatchSchema(BaseModel): pass
    class PatternDetail(BaseModel): pass
    class PatternAnalysisResult(BaseModel): pass
    class PatternAnalysisBatchSchema(BaseModel): pass
    class ClauseAnalysis(BaseModel): pass
    class FilteringMetadata(BaseModel): pass

# --- Constants ---
# Ensure consistency with preprocessing.py if possible
PROMPT_DIR = Path(__file__).parent / "prompts"
CORRECTION_PROMPT_FILE = PROMPT_DIR / "batch_correction_prompt.txt"
ACCURACY_PROMPT_FILE = PROMPT_DIR / "batch_accuracy_prompt.txt"
PATTERN_ANALYSIS_PROMPT_FILE = PROMPT_DIR / "batch_pattern_analysis_prompt.txt" 
FILTERING_PROMPT_FILE = PROMPT_DIR / "batch_filtering_prompt.txt" # Add filtering prompt file
# Consider making the model name configurable or reading from env
MODEL_NAME = os.getenv("ANALYZER_MODEL_NAME", "gemini-1.5-flash") # Match preprocessing or use env var

# Initialize global chain variables to None
# These will be populated by the _initialize_* functions
correction_chain = None
accuracy_chain = None
pattern_analysis_chain = None
filtering_chain = None

# Load environment variables (e.g., for API keys)
load_dotenv()

# Define the expected structure for each item in the LLM's batch output
class ClauseCorrectionResult(BaseModel):
    clause_id: str = Field(..., description="Must match the clause_id from the input.")
    corrected_clause_text: str = Field(..., description="The fully corrected version of the clause text.")

# Get the specific logger configured in logging.conf
logger = logging.getLogger("analyzer_service")

# --- Constants for Scoring --- #
# Complexity
MAX_CLAUSE_DENSITY = 2.5
MAX_MLC = 12.0
CLAUSE_WEIGHT = 0.6
MLC_WEIGHT = 0.4
# Accuracy
SEVERITY_WEIGHTS = {"critical": 0.4, "moderate": 0.2, "minor": 0.1}
LAMBDA_FACTOR = 1.2
# --- End Constants --- #

# Lexical complexity resources
WORD_LIST_DIR = Path(__file__).parent.parent / "word_lists"
NGSL_FILE = WORD_LIST_DIR / "NGSL_1.2_stats.csv"
NAWL_FILE = WORD_LIST_DIR / "NAWL_1.2_stats.csv"

def _load_ngsl():
    freq = {}
    with open(NGSL_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row['Lemma'].lower()
            try:
                adj_freq = float(row.get('Adjusted Frequency per Million (U)', 0.0))
            except ValueError:
                adj_freq = 0.0
            freq[lemma] = adj_freq
    return freq

def _load_nawl():
    ranks = {}
    with open(NAWL_FILE, newline='', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            # Defensive: handle BOM or whitespace in header
            keys = list(row.keys())
            word_key = keys[0].lstrip('\ufeff').strip() if keys else 'Word'
            word = row[word_key].lower()
            try:
                rank_val = int(row.get('Rank', 0))
            except ValueError:
                rank_val = 0
            ranks[word] = rank_val
    return ranks

NGSL_FREQ = _load_ngsl()
NAWL_RANKS = _load_nawl()
MAX_NGSL_FREQ = max(NGSL_FREQ.values()) if NGSL_FREQ else 1.0
MAX_NAWL_RANK = max(NAWL_RANKS.values()) if NAWL_RANKS else 1.0

def calculate_lexical_complexity(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    scores = []
    for tok in tokens:
        if tok in NGSL_FREQ:
            scores.append(0.0)
        elif tok in NAWL_RANKS:
            scores.append(1.0)
        else:
            scores.append(0.5)
    return sum(scores) / len(scores)

def _format_context(context: Optional[List[ContextUtterance]]) -> str:
    """Helper function to format context list into a simple string."""
    if not context:
        return "No context provided."
    return "\n".join([f"{utt.speaker}: {utt.text}" for utt in context])

async def _initialize_correction_chain():
    try:
        # Load prompt from file
        logger.debug(f"Loading correction prompt from: {CORRECTION_PROMPT_FILE}")
        with open(CORRECTION_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        # Ensure the template uses the correct single input variable for the batch
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["batch_clauses_json"] # <--- Use single variable
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1) # Adjust temp as needed

        # Initialize Parser (targeting a list of ClauseCorrectionResult)
        parser = StrOutputParser() # Instantiate StrOutputParser

        # Create the correction_chain
        correction_chain = prompt | llm | parser

        # Return the successfully created chain
        return correction_chain

    except Exception as e:
        logger.error(f"Unexpected error initializing correction chain: {e}", exc_info=True)
        return None # Ensure chain is None if init fails

async def _run_correction_chain_on_batch(batch: List[PreprocessedASUnit]):
    """Runs the correction chain on a batch of PreprocessedASUnit objects using a single API call.

    Args:
        batch: A list of PreprocessedASUnit objects.

    Returns:
        The original list of PreprocessedASUnit objects, potentially modified in-place
        with corrected clause text.
    """
    if not batch:
        logger.info("[Correction Chain] Empty batch received, skipping.")
        return batch

    # 1. Extract all clauses and create a map for quick lookup
    all_clauses: List[AlignedClause] = []
    clause_map: Dict[str, AlignedClause] = {}
    chain_input_list = [] # Prepare the list for JSON serialization here

    for unit in batch:
        context_str = _format_context(unit.context) # Get context from the unit
        if unit.clauses:
            for clause in unit.clauses:
                if clause.clause_text: # Only include clauses with text
                    all_clauses.append(clause)
                    clause_map[clause.clause_id] = clause
                    # Add clause details along with the UNIT's context
                    chain_input_list.append({
                        "clause_id": clause.clause_id,
                        "clause_text": clause.clause_text,
                        "context": context_str # Use the unit's context
                    })

    if not chain_input_list: # Check if any valid clauses were found and added
        logger.info("[Correction Chain] No clauses with text found in the batch units, skipping.")
        return batch

    logger.info(f"[Correction Chain] Preparing batch of {len(all_clauses)} clauses for single API call correction.")

    # 2. Prepare input for the chain: serialize the list of clause data to a JSON string
    try:
        batch_clauses_json_string = json.dumps(chain_input_list, ensure_ascii=False, indent=2)
    except TypeError as e:
        logger.error(f"[Correction Chain] Failed to serialize batch clauses to JSON: {e}", exc_info=True)
        return batch # Cannot proceed without valid JSON input

    single_input_dict = {"batch_clauses_json": batch_clauses_json_string}

    # 3. Initialize the correction chain
    chain = await _initialize_correction_chain()
    if chain is None:
        logger.error("[Correction Chain] Failed to initialize, skipping correction.")
        return batch
    logger.info("[Correction Chain] Correction chain initialized for single call.")

    # 4. Invoke the chain with the single JSON string input
    logger.info(f"[Correction Chain] Invoking chain with serialized batch...")
    start_time = time.time()
    try:
        # The chain now returns a single raw string (hopefully a JSON list)
        raw_result: str = await chain.ainvoke(single_input_dict) # Use ainvoke
        duration = time.time() - start_time
        logger.info(f"[Correction Chain] Chain invocation completed in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"[Correction Chain] Error during correction chain invocation: {e}", exc_info=True)
        return batch # Return original batch on chain error

    # 5. Robust Parsing and Validation of the Single Raw Result String
    parsed_corrected_clauses: List[ClauseCorrectionResult] = []
    try:
        # Clean potential markdown fences
        cleaned_result = raw_result.strip().removeprefix("```json").removesuffix("```").strip()
        
        # Attempt JSON parsing - expecting a list
        parsed_data = json.loads(cleaned_result)
        
        if isinstance(parsed_data, list):
            # LLM returned a list as expected, wrap it for validation
            structured_data = {"corrected_clauses": parsed_data}
            logger.debug(f"[Correction Chain] LLM returned a list. Wrapping for validation.")
            
            # Validate the structured data against the Pydantic schema
            validated_batch = CorrectionBatchSchema.model_validate(structured_data)
            parsed_corrected_clauses = validated_batch.corrected_clauses # Extract the list
            logger.info(f"[Correction Chain] Successfully parsed and validated corrections for {len(parsed_corrected_clauses)} clauses.")
        else:
            # LLM did not return a list as expected by the prompt
            logger.warning(f"[Correction Chain] Parsed JSON from LLM is not a list (type: {type(parsed_data)}). Expected a list based on prompt. Skipping correction application.")
            # No clauses are added to parsed_corrected_clauses

    except json.JSONDecodeError:
        logger.warning(f"[Correction Chain] Failed to decode JSON from LLM output. Raw output: '{raw_result}'")
    except (ValidationError, TypeError) as e: # Catch Pydantic validation errors
        logger.warning(f"[Correction Chain] Failed to validate LLM output against CorrectionBatchSchema. Error: {e}. Attempted structured data: '{structured_data}'")
    except Exception as e:
        logger.error(f"[Correction Chain] Unexpected error parsing result: {e}", exc_info=True)
        
    # 6. Update the original clauses with the corrected text
    processed_ids = set()
    if parsed_corrected_clauses: # Only proceed if we successfully parsed some clauses
        for correction in parsed_corrected_clauses:
            clause_id = correction.clause_id
            if clause_id in clause_map:
                original_clause = clause_map[clause_id]
                original_clause.corrected_clause_text = correction.corrected_clause_text
                processed_ids.add(clause_id)
            else:
                logger.warning(f"Received result for unknown clause_id '{clause_id}' from correction chain.")

        # Check if any input clauses were not processed
        unprocessed_ids = set(clause_map.keys()) - processed_ids
        if unprocessed_ids:
            logger.warning(f"Could not find matching results for clause IDs: {unprocessed_ids}")
    else:
        logger.warning("[Correction Chain] No corrected clauses were successfully parsed. Skipping update step.")

    logger.info(f"[Correction Chain] Finished correction chain processing for the batch (single call).")

    # Return the original list of AS Units, potentially modified in-place
    return batch

async def _initialize_accuracy_chain():
    """Initializes the LangChain chain for batch accuracy analysis (error identification)."""
    try:
        logger.debug(f"Loading accuracy prompt from: {ACCURACY_PROMPT_FILE}")
        with open(ACCURACY_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

        # Ensure the template uses the correct single input variable for the batch
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["batch_clauses_json"] # Single input variable for the JSON blob
        )

        # Initialize LLM (use the same model as correction for consistency, or configure differently)
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1) # Low temp for structured output

        # Use StrOutputParser as the LLM returns a JSON string
        parser = StrOutputParser()

        # Create the accuracy_chain
        accuracy_chain = prompt | llm | parser

        logger.info("Accuracy analysis chain initialized successfully.")
        return accuracy_chain

    except FileNotFoundError:
        logger.error(f"Accuracy prompt file not found: {ACCURACY_PROMPT_FILE}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing accuracy chain: {e}", exc_info=True)
        return None

async def _run_accuracy_analysis_on_batch(batch: List[PreprocessedASUnit]):
    """Runs the accuracy analysis chain on a batch to identify errors.

    Compares original_clause_text and corrected_clause_text to populate
    the errors_found list in each AlignedClause.

    Args:
        batch: List of PreprocessedASUnit objects (should have corrected_clause_text populated).

    Returns:
        The original list of PreprocessedASUnit objects, potentially modified in-place
        with populated errors_found lists.
    """
    if not batch:
        logger.info("[Accuracy Chain] Empty batch received, skipping.")
        return batch

    # 1. Prepare input: Collect clauses needing analysis and map for lookup
    clauses_for_analysis = []
    clause_map: Dict[str, AlignedClause] = {}

    for unit in batch:
        if unit.clauses:
            for clause in unit.clauses:
                # Only analyze if there's a correction and it's different from original
                if clause.corrected_clause_text and clause.clause_text != clause.corrected_clause_text:
                    clauses_for_analysis.append({
                        "clause_id": clause.clause_id,
                        "original_clause_text": clause.clause_text,
                        "corrected_clause_text": clause.corrected_clause_text
                    })
                    clause_map[clause.clause_id] = clause
                else:
                    # Ensure errors_found is initialized as empty if no analysis needed
                    if not hasattr(clause, 'errors_found') or clause.errors_found is None:
                         clause.errors_found = []

    if not clauses_for_analysis:
        logger.info("[Accuracy Chain] No clauses require error analysis in this batch.")
        # Ensure all clauses have errors_found initialized
        for unit in batch:
            if unit.clauses:
                for clause in unit.clauses:
                    if not hasattr(clause, 'errors_found') or clause.errors_found is None:
                        clause.errors_found = []
        return batch

    logger.info(f"[Accuracy Chain] Preparing batch of {len(clauses_for_analysis)} clauses for error analysis.")

    # 2. Serialize the list to a JSON string for the single API call
    try:
        batch_clauses_json_string = json.dumps(clauses_for_analysis, ensure_ascii=False, indent=2)
    except TypeError as e:
        logger.error(f"[Accuracy Chain] Failed to serialize batch clauses to JSON: {e}", exc_info=True)
        return batch # Cannot proceed without valid JSON input

    single_input_dict = {"batch_clauses_json": batch_clauses_json_string}

    # 3. Initialize the accuracy analysis chain
    chain = await _initialize_accuracy_chain()
    if chain is None:
        logger.error("[Accuracy Chain] Failed to initialize, skipping error analysis.")
        return batch
    logger.info("[Accuracy Chain] Accuracy chain initialized.")

    # 4. Invoke the chain
    logger.info(f"[Accuracy Chain] Invoking chain with serialized batch...")
    start_time = time.time()
    try:
        # Expecting a single raw JSON string as output
        raw_result: str = await chain.ainvoke(single_input_dict) # Use ainvoke
        duration = time.time() - start_time
        logger.info(f"[Accuracy Chain] Chain invocation completed in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"[Accuracy Chain] Error during accuracy chain invocation: {e}", exc_info=True)
        return batch # Return original batch on chain error

    # 5. Parse and Validate the LLM's JSON output
    parsed_analyses: List[AccuracyAnalysisResult] = []
    try:
        # Clean potential markdown fences
        cleaned_result = raw_result.strip().removeprefix("```json").removesuffix("```").strip()

        # Attempt JSON parsing - expecting the structure defined in AccuracyBatchSchema
        parsed_data = json.loads(cleaned_result)

        # Validate the entire structure using the batch schema
        validated_batch = AccuracyBatchSchema.model_validate(parsed_data)
        parsed_analyses = validated_batch.error_analyses # Extract the list of results
        logger.info(f"[Accuracy Chain] Successfully parsed and validated error analyses for {len(parsed_analyses)} clauses.")

    except json.JSONDecodeError:
        logger.warning(f"[Accuracy Chain] Failed to decode JSON from LLM output. Raw output: '{raw_result}'")
    except (ValidationError, TypeError) as e:
        logger.warning(f"[Accuracy Chain] Failed to validate LLM output against AccuracyBatchSchema. Error: {e}. Raw output: '{raw_result}'")
    except Exception as e:
        logger.error(f"[Accuracy Chain] Unexpected error parsing result: {e}. Raw output: '{raw_result}'", exc_info=True)

    # 6. Update the original clauses with the identified errors
    processed_ids = set()
    if parsed_analyses: # Only proceed if parsing/validation succeeded
        logger.info(f"[Accuracy Chain] Updating {len(parsed_analyses)} clauses with identified errors.")
        for analysis_result in parsed_analyses:
            clause_id = analysis_result.clause_id
            if clause_id in clause_map:
                original_clause = clause_map[clause_id]
                # Directly assign the validated list of ErrorDetail objects
                original_clause.errors_found = analysis_result.errors_found
                processed_ids.add(clause_id)
            else:
                logger.warning(f"[Accuracy Chain] Received result for unknown clause_id '{clause_id}' from accuracy chain.")

        # Check for clauses sent but not returned
        unprocessed_ids = set(clause_map.keys()) - processed_ids
        if unprocessed_ids:
            logger.warning(f"[Accuracy Chain] Did not receive error analysis results for clause IDs: {unprocessed_ids}")
    else:
        logger.warning("[Accuracy Chain] No error analyses were successfully parsed/validated. Skipping update step.")
        # Ensure clauses that were intended for analysis but failed have empty lists
        for clause_id in clause_map:
             if clause_id not in processed_ids:
                 clause_map[clause_id].errors_found = []

    return batch

async def _initialize_pattern_analysis_chain():
    """Initializes the LangChain components for the batch pattern analysis chain."""
    try:
        logger.debug(f"Loading pattern analysis prompt from: {PATTERN_ANALYSIS_PROMPT_FILE}")
        with open(PATTERN_ANALYSIS_PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

        # Ensure the template uses the correct single input variable for the batch
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["batch_clauses_json"] # Single input variable for the JSON blob
        )

        # Initialize LLM (use the same model as correction for consistency, or configure differently)
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1) # Low temp for structured output

        # Use StrOutputParser as the LLM returns a JSON string
        parser = StrOutputParser()

        # Create the pattern_analysis_chain
        pattern_analysis_chain = prompt | llm | parser

        logger.info("Pattern analysis chain initialized successfully.")
        return pattern_analysis_chain

    except FileNotFoundError:
        logger.error(f"Pattern analysis prompt file not found: {PATTERN_ANALYSIS_PROMPT_FILE}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing pattern analysis chain: {e}", exc_info=True)
        return None

async def _run_pattern_analysis_on_batch(batch: List[PreprocessedASUnit]):
    """Runs the pattern analysis chain on a batch to identify formulaic patterns.

    Analyzes the corrected_clause_text to populate the clause_pattern_analysis list
    in each relevant AlignedClause.

    Args:
        batch: List of PreprocessedASUnit objects (should have corrected_clause_text populated).

    Returns:
        The original list of PreprocessedASUnit objects, potentially modified in-place
        with populated clause_pattern_analysis lists.
    """
    if not batch:
        logger.info("[Pattern Chain] Empty batch received, skipping.")
        return batch

    # 1. Prepare input: Collect clauses needing analysis and map for lookup
    clauses_for_analysis = []
    clause_map: Dict[str, AlignedClause] = {}

    for unit in batch:
        if unit.clauses:
            for clause in unit.clauses:
                # Analyze if corrected text exists and is not empty
                if clause.corrected_clause_text:
                    clauses_for_analysis.append({
                        "clause_id": clause.clause_id,
                        "corrected_clause_text": clause.corrected_clause_text
                    })
                    clause_map[clause.clause_id] = clause
                else:
                    # Ensure clause_pattern_analysis is initialized if no analysis needed
                    if not hasattr(clause, 'clause_pattern_analysis') or clause.clause_pattern_analysis is None:
                         clause.clause_pattern_analysis = []

    if not clauses_for_analysis:
        logger.info("[Pattern Chain] No clauses require pattern analysis in this batch.")
        # Ensure all clauses have pattern analysis initialized
        for unit in batch:
            if unit.clauses:
                for clause in unit.clauses:
                    if not hasattr(clause, 'clause_pattern_analysis') or clause.clause_pattern_analysis is None:
                        clause.clause_pattern_analysis = []
        return batch

    logger.info(f"[Pattern Chain] Preparing batch of {len(clauses_for_analysis)} clauses for pattern analysis.")

    # 2. Serialize the list to a JSON string for the single API call
    try:
        batch_clauses_json_string = json.dumps(clauses_for_analysis, ensure_ascii=False, indent=2)
    except TypeError as e:
        logger.error(f"[Pattern Chain] Failed to serialize batch clauses to JSON: {e}", exc_info=True)
        return batch # Cannot proceed without valid JSON input

    single_input_dict = {"batch_clauses_json": batch_clauses_json_string}

    # 3. Initialize the pattern analysis chain
    chain = await _initialize_pattern_analysis_chain()
    if chain is None:
        logger.error("[Pattern Chain] Failed to initialize, skipping pattern analysis.")
        return batch
    logger.info("[Pattern Chain] Pattern analysis chain initialized.")

    # 4. Invoke the chain
    logger.info(f"[Pattern Chain] Invoking chain with serialized batch...")
    start_time = time.time()
    try:
        # Expecting a single raw JSON string as output
        raw_result: str = await chain.ainvoke(single_input_dict) # Use ainvoke
        duration = time.time() - start_time
        logger.info(f"[Pattern Chain] Chain invocation completed in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"[Pattern Chain] Error during pattern analysis chain invocation: {e}", exc_info=True)
        return batch # Return original batch on chain error

    # 5. Parse and Validate the LLM's JSON output
    parsed_analyses: List[PatternAnalysisResult] = []
    try:
        # Clean potential markdown fences
        cleaned_result = raw_result.strip().removeprefix("```json").removesuffix("```").strip()

        # Attempt JSON parsing - expecting the structure defined in PatternAnalysisBatchSchema
        parsed_data = json.loads(cleaned_result)

        # Validate the entire structure using the batch schema
        validated_batch = PatternAnalysisBatchSchema.model_validate(parsed_data)
        parsed_analyses = validated_batch.pattern_analyses # Extract the list of results
        logger.info(f"[Pattern Chain] Successfully parsed and validated pattern analyses for {len(parsed_analyses)} clauses.")

    except json.JSONDecodeError:
        logger.warning(f"[Pattern Chain] Failed to decode JSON from LLM output. Raw output: '{raw_result}'")
    except (ValidationError, TypeError) as e:
        logger.warning(f"[Pattern Chain] Failed to validate LLM output against PatternAnalysisBatchSchema. Error: {e}. Raw output: '{raw_result}'")
    except Exception as e:
        logger.error(f"[Pattern Chain] Unexpected error parsing result: {e}. Raw output: '{raw_result}'", exc_info=True)

    # 6. Update the original clauses with the identified patterns
    processed_ids = set()
    if parsed_analyses: # Only proceed if parsing/validation succeeded
        logger.info(f"[Pattern Chain] Updating {len(parsed_analyses)} clauses with identified patterns.")
        for analysis_result in parsed_analyses:
            clause_id = analysis_result.clause_id
            if clause_id in clause_map:
                # Ensure the target field exists and is a list before extending
                if not hasattr(clause_map[clause_id], 'clause_pattern_analysis') or clause_map[clause_id].clause_pattern_analysis is None:
                    clause_map[clause_id].clause_pattern_analysis = []

                # Append patterns (list of PatternDetail objects)
                # Pydantic validation already ensures analysis_result.patterns is a List[PatternDetail]
                clause_map[clause_id].clause_pattern_analysis.extend(analysis_result.patterns)
                processed_ids.add(clause_id)
            else:
                logger.warning(f"[Pattern Chain] Received analysis for unknown clause_id: {clause_id}")
    else:
        logger.warning("[Pattern Chain] No valid pattern analyses were parsed from the LLM output.")

    # Log clauses that were expected but not returned/processed
    expected_ids = set(clause_map.keys())
    missing_ids = expected_ids - processed_ids
    if missing_ids:
        logger.warning(f"[Pattern Chain] Did not receive/process pattern analysis for {len(missing_ids)} expected clauses: {missing_ids}")

    # Ensure all clauses in the original batch have the field initialized, even if empty
    for unit in batch:
        if unit.clauses:
            for clause in unit.clauses:
                if not hasattr(clause, 'clause_pattern_analysis') or clause.clause_pattern_analysis is None:
                    clause.clause_pattern_analysis = []

    return batch

async def _initialize_filtering_chain():
    """Initializes the LangChain components for the batch AI Filtering chain."""
    global filtering_chain
    if filtering_chain is None:
        logger.debug("Initializing AI Filtering Chain...")
        try:
            # Load prompt from file
            logger.debug(f"Loading filtering prompt from: {FILTERING_PROMPT_FILE}")
            with open(FILTERING_PROMPT_FILE, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            # Ensure the template uses the correct single input variable for the batch
            prompt = PromptTemplate(
                template=prompt_text,
                input_variables=["batch_filtering_input"] # Single input variable for the JSON blob
            )

            # Initialize LLM
            llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1) # Low temp for structured output

            # Use StrOutputParser as the LLM returns a JSON string
            parser = StrOutputParser()

            # Create the filtering_chain
            filtering_chain = prompt | llm | parser
            logger.info("AI Filtering chain initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Filtering chain: {e}", exc_info=True)
            # Ensure chain remains None if init fails
            filtering_chain = None

async def _parse_filtering_output(raw_output: str, expected_units: int) -> List[Optional[FilteringMetadata]]:
    """Parses the raw string output from the filtering chain."""
    parsed_results: List[Optional[FilteringMetadata]] = [None] * expected_units
    logger.debug(f"Raw filtering output to parse (first 500 chars): {raw_output[:500]}")
    
    # Strip Markdown fences if present
    processed_output = raw_output.strip()
    if processed_output.startswith("```json"):
        processed_output = processed_output[len("```json"):].strip()
    if processed_output.startswith("```"):
         processed_output = processed_output[len("```"):].strip()
    if processed_output.endswith("```"):
        processed_output = processed_output[:-len("```")].strip()

    logger.debug(f"Processed filtering output for JSON parsing: {processed_output[:500]}")

    try:
        # Attempt to find the JSON list within the raw output if necessary
        # Simple case: Assume raw_output IS the JSON list string
        data = json.loads(processed_output) # Use processed_output
        if isinstance(data, list):
            if len(data) != expected_units:
                 logger.warning(f"Filtering output count mismatch: Expected {expected_units}, Got {len(data)}. Assigning based on received items.")

            validated_count = 0
            for i, item in enumerate(data):
                if i >= expected_units: # Stop if we have more results than expected inputs
                    logger.warning(f"Filtering output had more items ({len(data)}) than expected ({expected_units}). Ignoring extra items.")
                    break
                if isinstance(item, dict):
                    try:
                        parsed_results[i] = FilteringMetadata(**item)
                        validated_count += 1
                    except ValidationError as e:
                        logger.warning(f"Filtering validation failed for item {i}: {e}. Details: {item}. Setting to None.")
                    except TypeError as e:
                         logger.warning(f"Filtering type error during validation for item {i}: {e}. Details: {item}. Setting to None.")
                else:
                    logger.warning(f"Filtering output item {i} is not a dictionary (Type: {type(item)}). Setting to None.")
            logger.info(f"Successfully parsed and validated {validated_count}/{len(data)} filtering results (Expected {expected_units}).")
        else:
            logger.error("Filtering output JSON is not a list.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode filtering JSON: {e}. Raw output: '{raw_output}'")
    except Exception as e:
        logger.error(f"Error processing filtering output: {e}", exc_info=True)

    # Final check: Ensure list length matches expected_units
    if len(parsed_results) != expected_units:
        logger.error(f"Internal Error: Filtering result list length {len(parsed_results)} != expected {expected_units} after parsing. Fixing length.")
        parsed_results = (parsed_results + [None] * expected_units)[:expected_units]

    return parsed_results

async def run_analysis_batch(preprocessed_batch: List[PreprocessedASUnit]) -> List[Dict[str, Any]]: # Keep type hint for consistency
    """Runs the full analysis pipeline (correction, accuracy, patterns, scoring, filtering) on a batch.

    Args:
        preprocessed_batch: A list of PreprocessedASUnit objects.

    Returns:
        A list of dictionaries representing the MainAnalysisOutput structure.
        (Note: Actual return is List[Dict] for JSON serialization compatibility, 
         despite the List[MainAnalysisOutput] type hint).
    """
    start_time = time.time()
    logger.info(f"[Analysis Module] Starting analysis for batch of {len(preprocessed_batch)} units.")

    # Initialize parsers (can be done at module level)
    try:
        correction_parser = PydanticOutputParser(pydantic_object=CorrectionBatchSchema)
    except Exception as e:
        logger.error(f"Failed to initialize correction parser: {e}", exc_info=True)

    # 1. Correction Analysis
    # Modifies the PreprocessedASUnit objects in-place
    logger.info("Running correction analysis...")
    await _run_correction_chain_on_batch(preprocessed_batch)
    logger.info("Correction analysis finished.")

    # 2 & 3. Run Accuracy and Pattern Analysis Concurrently
    logger.info("Running accuracy and pattern analysis concurrently...")
    await asyncio.gather(
        _run_accuracy_analysis_on_batch(preprocessed_batch),
        _run_pattern_analysis_on_batch(preprocessed_batch)
    )
    logger.info("Accuracy and pattern analysis finished.")

    # --- Start: Add Korean Vocabulary Error Logic ---
    logger.info("Checking for Korean vocabulary usage...")
    for unit in preprocessed_batch:
        if unit.clauses:
            for clause in unit.clauses:
                if clause.is_korean:
                    logger.debug(f"Clause {clause.clause_id} is Korean. Type: {clause.original_clause_type}")
                    # Determine severity based on original_clause_type
                    severity = Severity.CRITICAL # Default to critical
                    if clause.original_clause_type:
                        clause_type_lower = clause.original_clause_type.lower()
                        if clause_type_lower == 'word':
                            severity = Severity.MINOR
                        elif clause_type_lower == 'collocation':
                            severity = Severity.MODERATE
                        # 'phrase' or other types remain critical

                    # Create the Korean Vocabulary error detail
                    # Use placeholders for error/correction as model requires strings
                    korean_error = ErrorDetail(
                        category="Korean Vocabulary",
                        severity=severity.value, # Use the enum value
                        error="Korean segment used instead of English.",
                        correction="[Use English equivalent]" # Placeholder
                    )

                    # Ensure errors_found list exists before appending
                    if not hasattr(clause, 'errors_found') or clause.errors_found is None:
                        clause.errors_found = [] # Initialize if it was skipped by accuracy chain

                    # Append the error
                    clause.errors_found.append(korean_error)
                    logger.debug(f"Appended Korean Vocabulary error to clause {clause.clause_id} with severity {severity.value}")
    logger.info("Finished checking for Korean vocabulary usage.")
    # --- End: Add Korean Vocabulary Error Logic ---

    # Initialize the filtering chain (if not already done)
    await _initialize_filtering_chain()

    # 4. Run AI Filtering Chain
    filtering_results: List[Optional[FilteringMetadata]] = [None] * len(preprocessed_batch)

    if filtering_chain: # Check if the chain object exists (was initialized successfully)
        logger.info("Preparing input for AI Filtering chain...")
        batch_filtering_input = []
        for unit in preprocessed_batch:
            # Extract errors from all clauses for this unit
            all_errors_for_unit = []
            for clause in unit.clauses:
                 # Convert ErrorDetail objects to dictionaries for JSON serialization
                 all_errors_for_unit.extend([err.model_dump() for err in clause.errors_found])

            unit_input = {
                "as_unit_id": unit.as_unit_id,
                "original_text": unit.aligned_original_text,
                # Fix: Assemble corrected text by joining corrected text from each clause
                "corrected_text": " ".join(clause.corrected_clause_text for clause in unit.clauses if clause.corrected_clause_text),
                "errors_found": all_errors_for_unit
            }
            batch_filtering_input.append(unit_input)

        if batch_filtering_input:
            try:
                input_json_str = json.dumps(batch_filtering_input, ensure_ascii=False)
                logger.info(f"Invoking AI Filtering chain for {len(batch_filtering_input)} units...")
                start_filter_time = time.time()
                raw_filtering_output = await filtering_chain.ainvoke({"batch_filtering_input": input_json_str})
                end_filter_time = time.time()
                logger.info(f"AI Filtering chain invocation finished in {end_filter_time - start_filter_time:.2f}s.")

                # Parse the raw output string
                filtering_results = await _parse_filtering_output(raw_filtering_output, len(preprocessed_batch))
                logger.info(f"Parsed {sum(1 for r in filtering_results if r)} filtering results.")

            except Exception as e:
                logger.error(f"Error during AI Filtering chain invocation or parsing: {e}", exc_info=True)
                # filtering_results remains list of Nones
        else:
            logger.info("No input generated for AI Filtering chain.")
    else:
        logger.warning("AI Filtering chain not initialized. Skipping filtering step.")


    # 5. Assemble MainAnalysisOutput objects, Calculate Scores, Assign Filtering, and Convert to Dict
    analysis_results_dicts: List[Dict[str, Any]] = [] # List to hold final dicts
    for index, unit in enumerate(preprocessed_batch):
        logger.debug(f"Assembling final analysis output for unit {unit.as_unit_id}")

        # Map AlignedClause to ClauseAnalysis, ensuring errors are serializable
        clause_analyses = []
        for clause in unit.clauses:
            # Ensure errors_found contains ErrorDetail objects for scoring but dump for final output
            clause_analysis_obj = ClauseAnalysis(
                clause_id=clause.clause_id,
                clause_text=clause.clause_text,
                corrected_clause_text=clause.corrected_clause_text,
                errors_found=clause.errors_found, # Keep as objects for scoring
                clause_pattern_analysis=clause.clause_pattern_analysis
            )
            clause_analyses.append(clause_analysis_obj)

        # Create the MainAnalysisOutput object
        analysis_output_obj = MainAnalysisOutput(
            as_unit_id=unit.as_unit_id,
            original_text=unit.aligned_original_text, # Use the aligned original text segment
            # Fix: Assemble corrected text by joining corrected text from each clause
            corrected_text=" ".join(clause.corrected_clause_text for clause in unit.clauses if clause.corrected_clause_text),
            complexity_score=0.0, # Placeholder, calculated next
            accuracy_score=0.0,   # Placeholder, calculated next
            clauses=clause_analyses, # Use the ClauseAnalysis objects created above
            context=unit.context, # Pass context along
            # filtering_metadata will be assigned next
            filtering_metadata=None # Placeholder
        )

        # 6. Calculate and add scores to the object
        _calculate_scores(analysis_output_obj) # Pass the OBJECT

        # 7. Assign Filtering Metadata from parsed results
        if index < len(filtering_results):
            analysis_output_obj.filtering_metadata = filtering_results[index]
        else:
            logger.warning(f"Filtering result missing for index {index}, unit {unit.as_unit_id}")
            analysis_output_obj.filtering_metadata = None

        # 8. Convert the scored object to a dictionary for JSON serialization
        try:
            # exclude_none=True helps keep the output clean
            analysis_output_dict = analysis_output_obj.model_dump(exclude_none=True)
            analysis_results_dicts.append(analysis_output_dict)
        except Exception as e:
            logger.error(f"Error dumping MainAnalysisOutput model for unit {unit.as_unit_id}: {e}", exc_info=True)
            # Append a basic error dict or skip?
            analysis_results_dicts.append({
                "as_unit_id": unit.as_unit_id,
                "error": "Failed to serialize analysis results",
                "detail": str(e)
            })

    end_time = time.time()
    logger.info(f"[Analysis Module] Finished processing batch, returning {len(analysis_results_dicts)} analysis result dictionaries. Time: {end_time - start_time:.2f}s")
    return analysis_results_dicts # Return the list of dictionaries

def _calculate_scores(analysis_output: MainAnalysisOutput):
    """Calculates complexity and accuracy scores and updates the MainAnalysisOutput object."""
    logger.debug(f"Calculating scores for unit {analysis_output.as_unit_id}")
    clauses = analysis_output.clauses
    total_clauses = len(clauses)

    # --- Complexity Score --- #
    if total_clauses > 0:
        # Calculate total words in the original AS unit text
        total_words_in_as_unit = len(analysis_output.original_text.split()) if analysis_output.original_text else 0

        if total_words_in_as_unit > 0:
            # Clause Density (relative to total words in AS unit)
            clause_density = float(total_clauses) / total_words_in_as_unit
            normalized_clause_density = min(clause_density / MAX_CLAUSE_DENSITY, 1.0)
        else:
            # Avoid division by zero if original_text is empty or has no words
            clause_density = 0.0
            normalized_clause_density = 0.0

        # Raw clause count feature
        MAX_CLAUSES = 5.0  # max expected clauses for normalization
        normalized_clause_count = min(float(total_clauses) / MAX_CLAUSES, 1.0)

        # Combine clause density and clause count equally
        clause_feature = 0.5 * normalized_clause_density + 0.5 * normalized_clause_count

        # Mean Length of Clause (MLC)
        total_words_in_clauses = 0
        for clause in clauses:
            # Use corrected text if available, otherwise original clause text
            text_to_count = clause.corrected_clause_text if clause.corrected_clause_text else clause.clause_text
            if text_to_count:
                total_words_in_clauses += len(text_to_count.split())
        mlc = float(total_words_in_clauses) / total_clauses # total_clauses is already checked > 0
        normalized_mlc = min(mlc / MAX_MLC, 1.0)

        # Lexical complexity component
        normalized_lexical_complexity = calculate_lexical_complexity(analysis_output.original_text or "")
        # Combine MLC and lexical complexity equally (50/50)
        composite_mlc = 0.5 * normalized_mlc + 0.5 * normalized_lexical_complexity

        complexity_score = (CLAUSE_WEIGHT * clause_feature) + (MLC_WEIGHT * composite_mlc)
    else:
        # Handle case with no clauses
        complexity_score = 0.0
        # Also zero out related metrics for clarity if needed
        clause_density = 0.0
        normalized_clause_density = 0.0
        mlc = 0.0
        normalized_mlc = 0.0

    analysis_output.complexity_score = round(complexity_score, 4) # Store score
    logger.debug(f"Unit {analysis_output.as_unit_id} Complexity Score: {analysis_output.complexity_score}")


    # --- Accuracy Score --- #
    all_errors: List[ErrorDetail] = []
    for clause in clauses:
        all_errors.extend(clause.errors_found)

    if all_errors:
        total_impact = 0.0 # Initialize here, only if needed
        for error in all_errors:
            # Access the attribute directly, not via .get()
            severity = error.severity.lower()
            weight = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS['minor'])
            total_impact += weight
        # Calculate score based on impact
        accuracy_score = math.exp(-LAMBDA_FACTOR * total_impact)
    else:
        # Handle the case where there are no errors
        total_impact = 0.0 # Explicitly zero impact
        accuracy_score = 1.0 # Perfect score if no errors

    analysis_output.accuracy_score = round(accuracy_score, 4) # Store score
    logger.debug(f"Unit {analysis_output.as_unit_id} Accuracy Score: {analysis_output.accuracy_score}")
