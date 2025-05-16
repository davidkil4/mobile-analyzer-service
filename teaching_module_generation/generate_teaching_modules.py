import os
import json
import logging
import os
import re
import sys
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
import time

# Import validation functionality from streamlined_validate_module.py
from teaching_module_generation.streamlined_validate_module import call_llm_for_problem_validation, call_llm_for_batch_problem_validation, ProblemValidationResult, ModuleValidationReport
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from teaching_module_generation.teaching_schemas import (
    InputRecommendation,
    InputContextTurn,
    InputErrorDetail,
    LLMExplanationOutput,
    LLMProblemsOutput,
    LLMProblem,
    TeachingModule,
    Problem,
    ProblemFeedback,
    LLMExplanation
)

# --- Configuration & Constants ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)

# Safety settings for Gemini API (adjust as needed)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.7, # Adjust for creativity vs. determinism
    # top_p=0.9,
    # top_k=40,
    # max_output_tokens=2048, # Default is often fine
    response_mime_type="application/json" # Critical for structured output
)

MODEL_NAME = "gemini-1.5-flash-latest" # Or your preferred model

# File Paths (relative to this script's location)
BASE_DIR = Path(__file__).parent
# Default paths (can be overridden by parameters to main())
DEFAULT_INPUT_DIR = BASE_DIR / "output_files" / "output_files" # As per user's latest message
DEFAULT_OUTPUT_DIR = BASE_DIR / "output_files" / "teaching_module_outputs_new"
PROMPT_DIR = BASE_DIR / "teaching_prompts"

INPUT_FILES_CONFIG = {
    "grammar": {
        "file_name": "grammar.json",
        "module_id_prefix": "grammar",
        "default_module_type": "CONVERSATIONAL_LOW" # Example default
    },
    "patterns": {
        "file_name": "patterns.json",
        "module_id_prefix": "pattern",
        "default_module_type": "DRILL_PRACTICE" # Example default
    },
    "vocab": {
        "file_name": "vocabulary_and_small_grammar.json",
        "module_id_prefix": "vocab",
        "default_module_type": "CONVERSATIONAL_LOW" # Example default
    },
    "korean_word": {
        "file_name": "output_files/output_files/korean_word_focused.json",
        "module_id_prefix": "korean_word",
        "default_module_type": "CONVERSATIONAL_LOW"
    },
    "korean_phrase": {
        "file_name": "output_files/output_files/korean_phrase_focused.json",
        "module_id_prefix": "korean_phrase",
        "default_module_type": "CONVERSATIONAL_LOW"
    }
}

PROMPT_PATHS = {
    "explanation": PROMPT_DIR / "explanation_prompt.txt",
    "korean_explanation": PROMPT_DIR / "korean_explanation_prompt.txt",
    "mcq": PROMPT_DIR / "mcq_revised_prompt.txt",
    "fill_blank": PROMPT_DIR / "fill_blank_prompt.txt",
    "dictation": PROMPT_DIR / "dictation_prompt.txt",
}

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- Helper Functions ---

def load_prompt_template(prompt_name: str) -> str:
    """Loads a prompt template from the specified file."""
    path = PROMPT_PATHS.get(prompt_name)
    if not path or not path.exists():
        logger.error(f"Prompt template file not found for: {prompt_name} at {path}")
        raise FileNotFoundError(f"Prompt template file not found for: {prompt_name} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def process_bracketed_dictation(target_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Processes dictation target_text containing one or more bracketed phrases.
    Returns (prompt_text, cleaned_target_text) or (None, None) if invalid.
    Example: "She [went] [to] the store." -> ("She ____ ____ the store.", "She went to the store.")
    """
    # Find all bracketed phrases
    bracketed_phrases = re.findall(r'\[(.*?)\]', target_text)
    if not bracketed_phrases:
        logger.warning(f"No brackets found in dictation target_text: {target_text}")
        return None, None
    
    # Log how many bracketed phrases were found
    if len(bracketed_phrases) > 1:
        logger.info(f"Found {len(bracketed_phrases)} bracketed phrases in dictation target_text: {target_text}")
    
    # Create prompt_text by replacing all bracketed phrases with blanks
    prompt_text = target_text
    for phrase in bracketed_phrases:
        prompt_text = prompt_text.replace(f'[{phrase}]', '____')
    
    # Remove all brackets for the clean target_text
    cleaned_target_text = target_text.replace('[', '').replace(']', '')
    
    return prompt_text, cleaned_target_text

async def call_gemini_api(
    prompt: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS
) -> Optional[str]:
    """Calls the Gemini API with retry logic and returns the text content."""
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )
    for attempt in range(max_retries):
        try:
            response = await model.generate_content_async(prompt)
            # Accessing the text content correctly based on Gemini API structure
            if response.parts:
                # Assuming the first part contains the desired text/JSON
                # Ensure it's text before trying to parse as JSON later
                candidate_text = response.text # Safest way to get combined text
                if candidate_text:
                    return candidate_text.strip()
            logger.warning(f"Gemini API returned no usable content. Attempt {attempt + 1}/{max_retries}. Full response: {response}")
            # If parts exist but no text, or other unexpected structure:
            # Consider logging response.prompt_feedback or specific error fields if available

        except Exception as e:
            logger.error(f"Error calling Gemini API (attempt {attempt + 1}/{max_retries}): {e}")
            if "RATE_LIMIT_EXCEEDED" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, genai.types.generation_types.StopCandidateException):
                 logger.info(f"Rate limit or specific API error encountered. Retrying in {retry_delay}s...")
                 await asyncio.sleep(retry_delay)
            elif attempt + 1 == max_retries:
                 logger.error("Max retries reached for Gemini API call.")
                 return None # Propagate failure after max retries
            else:
                # For other errors, retry immediately or after a short delay
                await asyncio.sleep(1)
        else: # If no exception, but no content found in 'if response.parts:' block
            if attempt + 1 < max_retries:
                logger.info(f"Retrying due to empty/unexpected response structure in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Max retries reached due to empty/unexpected API response.")
                return None
                
    return None

# --- Core Generation Functions ---

async def generate_explanation(
    input_data: InputRecommendation,
    prompt_template: str
) -> Optional[LLMExplanation]:
    """Generates explanation for a given input using the LLM."""
    # Defensive logging: dump input_data
    logging.error(f"[DEBUG] generate_explanation input_data: {input_data.model_dump_json(indent=2)}")
    # Build context dict
    context = {
        'original_utterance': input_data.original,
        'corrected_utterance': input_data.corrected,
        'reason': input_data.reason if input_data.reason is not None else "No specific reason provided.",
        'selected_error_type': getattr(input_data.selected_error, 'type', None),
        'selected_error_description': getattr(input_data.selected_error, 'description', None),
        'selected_error_correction': getattr(input_data.selected_error, 'correction', None),
        'conversation_context': "\n".join([f"{turn.speaker}: {turn.text}" for turn in input_data.context]) if input_data.context else "No conversation context available.",
        'utterance_json': input_data.model_dump_json(indent=2)
    }
    logging.error(f"[DEBUG] generate_explanation context dict: {json.dumps(context, ensure_ascii=False, indent=2)}")
    required_fields = ['original_utterance', 'corrected_utterance', 'reason', 'selected_error_type', 'selected_error_description', 'selected_error_correction']
    for field in required_fields:
        if context.get(field) is None:
            logging.error(f"[ERROR] Missing required field '{field}' in context for explanation. Skipping. Context: {json.dumps(context, ensure_ascii=False)}")
            return None
    # Try to add focus_type if present
    if hasattr(input_data, 'focus_type') and getattr(input_data, 'focus_type', None) is not None:
        context['focus_type'] = getattr(input_data, 'focus_type')
    else:
        logging.error(f"[ERROR] Missing 'focus_type' in input_data for explanation. Context: {json.dumps(context, ensure_ascii=False)}")
        return None
    try:
        prompt = prompt_template.format(**context)
    except KeyError as e:
        logging.error(f"[ERROR] KeyError formatting prompt: {e}. Context: {json.dumps(context, ensure_ascii=False)}")
        return None

    logger.info(f"Generating explanation for: {input_data.original[:50]}...")
    llm_response_text = await call_gemini_api(prompt)
    if not llm_response_text:
        logger.error("Failed to get response from LLM for explanation.")
        return None

    def robust_json_loads(s):
        import json
        import re
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.error(f"Initial JSON decode failed: {e}")
            # Try to fix common escape issues (e.g., bad backslashes)
            # Only replace single backslashes not part of valid escape sequences
            cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)
            try:
                return json.loads(cleaned)
            except Exception as e2:
                logger.error(f"Second JSON decode failed: {e2}")
                # Write to debug file for manual inspection
                with open("llm_failed_output_debug.json", "a", encoding="utf-8") as f:
                    f.write(s + "\n\n")
                return None

    # Try robust parse first
    parsed_json = robust_json_loads(llm_response_text)
    if parsed_json is None:
        logger.error(f"LLM Response could not be parsed as JSON, skipping.\n{llm_response_text}")
        return None
    try:
        parsed_output = LLMExplanationOutput(**parsed_json)
        return parsed_output.explanations
    except ValidationError as e:
        logger.error(f"Pydantic validation error for explanation LLM output: {e}")
        logger.error(f"LLM Response that failed validation:\n{llm_response_text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during explanation validation: {e}")
        logger.error(f"LLM Response (not valid JSON):\n{llm_response_text}")
        return None

async def generate_problems(
    input_data: InputRecommendation,
    problem_type: Literal["mcq", "fill_blank", "dictation"],
    num_problems: int,
    prompt_templates: Dict[str, str]
) -> List[Problem]: # Changed to return List[Problem] which includes problem_id
    """Generates a specified number of problems of a given type."""
    if problem_type not in prompt_templates:
        logger.error(f"Prompt template for problem type '{problem_type}' not found.")
        return []

    prompt_template = prompt_templates[problem_type]
    # Prepare prompt
    prompt = prompt_template.format(
        original_utterance=input_data.original,
        corrected_utterance=input_data.corrected,
        selected_error_type=input_data.selected_error.type,
        selected_error_description=input_data.selected_error.description,
        selected_error_correction=input_data.selected_error.correction,
        num_problems=num_problems,
        utterance_json=input_data.model_dump_json(indent=2) # Provide the full input data as JSON
        # context_dialogue="\n".join([f"{turn.speaker}: {turn.text}" for turn in input_data.context]) if input_data.context else "N/A"
    )

    logger.info(f"Generating {num_problems} '{problem_type}' problems for: {input_data.original[:50]}...")
    llm_response_text = await call_gemini_api(prompt)
    if not llm_response_text:
        logger.error(f"Failed to get response from LLM for {problem_type} problems.")
        return []

    final_problems: List[Problem] = []
    try:
        # LLM is expected to return JSON matching LLMProblemsOutput structure
        parsed_output = LLMProblemsOutput.model_validate_json(llm_response_text)
        
        for i, llm_problem in enumerate(parsed_output.problems):
            problem_id = f"{input_data.source_file_type}_{problem_type}_{input_data.utterance_index_in_file}_{i+1}"
            
            # Specific processing for dictation problems
            if llm_problem.type == "Dictation":
                if not llm_problem.target_text:
                    logger.warning(f"Dictation problem from LLM missing target_text. Skipping. Problem data: {llm_problem}")
                    continue
                
                prompt_text, cleaned_target_text = process_bracketed_dictation(llm_problem.target_text)
                if not prompt_text or not cleaned_target_text:
                    logger.warning(f"Failed to process bracketed dictation for target: {llm_problem.target_text}. Skipping problem.")
                    continue
                
                final_problems.append(Problem(
                    problem_id=problem_id,
                    type="Dictation",
                    prompt_text=prompt_text,
                    target_text=cleaned_target_text
                ))
            elif llm_problem.type == "MCQ" or llm_problem.type == "FillBlankChoice":
                # Basic validation that required fields are present for MCQ/FillBlank
                if not llm_problem.question or not llm_problem.options or not llm_problem.feedback:
                    logger.warning(f"{llm_problem.type} problem from LLM missing required fields (question, options, or feedback). Skipping. Problem data: {llm_problem}")
                    continue
                
                # Convert LLMProblemFeedback to ProblemFeedback
                problem_feedback_list = [
                    ProblemFeedback(option_text=f.option_text, is_correct=f.is_correct, explanation=f.explanation)
                    for f in llm_problem.feedback
                ]

                if llm_problem.type == "MCQ":
                    final_problems.append(Problem(
                        problem_id=problem_id,
                        type="MCQ",
                        question=llm_problem.question,
                        options=llm_problem.options,
                        feedback=problem_feedback_list
                    ))
                else: # FillBlankChoice
                    final_problems.append(Problem(
                        problem_id=problem_id,
                        type="FillBlankChoice",
                        question_template=llm_problem.question, # LLM prompt should guide it to use 'question' as 'question_template'
                        options=llm_problem.options,
                        feedback=problem_feedback_list
                    ))
            else:
                logger.warning(f"Unknown problem type from LLM: {llm_problem.type}. Skipping.")

    except ValidationError as e:
        logger.error(f"Pydantic validation error for {problem_type} LLM output: {e}")
        logger.error(f"LLM Response that failed validation:\n{llm_response_text}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {problem_type} LLM output: {e}")
        logger.error(f"LLM Response (not valid JSON):\n{llm_response_text}")
        
    return final_problems


# --- Main Orchestration ---

def load_and_parse_input_file(file_path: Path, source_file_type: str) -> List[InputRecommendation]:
    """Loads a JSON input file and parses its recommendations."""
    recommendations = []
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return []

    logger.info(f"Loading and parsing input file: {file_path} (source_file_type: {source_file_type})")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON from {file_path}")
            # Log the structure of the loaded data
            if isinstance(data, dict):
                logger.info(f"Data is a dictionary with keys: {list(data.keys())}")
                if 'analysis_zones' in data:
                    zones = data.get('analysis_zones', [])
                    logger.info(f"Found {len(zones)} analysis zones")
                    for i, zone in enumerate(zones):
                        zone_name = zone.get('zone_name', 'Unknown')
                        recs = zone.get('recommendations', [])
                        logger.info(f"Zone {i+1}: '{zone_name}' with {len(recs)} recommendations")
            elif isinstance(data, list):
                logger.info(f"Data is a list with {len(data)} items")
            else:
                logger.info(f"Data is of type: {type(data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return []

    # Handle different input formats
    utterance_idx_counter = 0
    
    # Check if data is a list (Korean format) or a dict with analysis_zones (standard format)
    if isinstance(data, list):
        # Korean format: direct list of utterances
        utterances_to_process = data
        logger.info(f"Detected Korean format (list). Found {len(utterances_to_process)} utterances to process.")
    else:
        # Standard format: {"analysis_zones": [{"recommendations": [...]}]}
        utterances_to_process = []
        for zone in data.get("analysis_zones", []):
            utterances_to_process.extend(zone.get("recommendations", []))
        logger.info(f"Detected standard format (analysis_zones). Found {len(utterances_to_process)} utterances to process.")
    
    # Process each utterance
    logger.info(f"Processing {len(utterances_to_process)} utterances from {file_path}")
    for i, rec_data in enumerate(utterances_to_process):
            try:
                # Extract selected_error from the 'errors' list or 'selected_error' field if present
                selected_error_data = rec_data.get("selected_error")
                if not selected_error_data:
                    # If 'selected_error' is not directly available, try to get it from 'errors'
                    # This logic might need adjustment based on actual data structure variance
                    # For now, assume the first error in 'errors' is the selected one if 'selected_error' is absent
                    if rec_data.get("errors"):
                        # Need to map fields if 'errors' structure differs, e.g. 'category' to 'type'
                        first_error = rec_data["errors"][0]
                        selected_error_data = {
                            "type": first_error.get("category", first_error.get("type")),
                            "severity": first_error.get("severity"),
                            "description": first_error.get("error", first_error.get("description")),
                            "correction": first_error.get("correction")
                        }
                    else:
                        logger.warning(f"Skipping recommendation due to missing 'selected_error' and 'errors' field: {rec_data.get('original')[:50]}...")
                        continue
                
                # Parse context
                input_context = []
                raw_context = rec_data.get("context", rec_data.get("preceding_context", []))
                if raw_context:
                    for turn_data in raw_context:
                        input_context.append(InputContextTurn(speaker=turn_data.get("speaker"), text=turn_data.get("text")))

                recommendations.append(InputRecommendation(
                    original=rec_data["original"],
                    corrected=rec_data["corrected"],
                    selected_error=InputErrorDetail(**selected_error_data),
                    context=input_context,
                    focus_type=rec_data.get("focus_type"),
                    reason=rec_data.get("reason"),
                    utterance_index_in_file=utterance_idx_counter,
                    source_file_type=source_file_type
                ))
                utterance_idx_counter += 1
            except ValidationError as e:
                logger.error(f"Pydantic validation error for recommendation: {e}. Data: {rec_data}")
            except KeyError as e:
                error_details = f"Caught KeyError in load_and_parse_input_file. Error Type: {type(e)}, Error Args: {e.args}, Key: '{e.args[0] if e.args else 'Unknown Key'}'. Offending rec_data: {json.dumps(rec_data, indent=2)}"
                logger.error(error_details)
                print(f"DEBUG STDERR: {error_details}", file=sys.stderr) # Explicit print to stderr
            except Exception as e:
                logger.error(f"Unexpected error parsing recommendation: {e}. Data: {rec_data}")
    return recommendations

async def validate_teaching_module(teaching_module: TeachingModule) -> Tuple[Optional[TeachingModule], ModuleValidationReport]:
    """
    Validates a teaching module using the validation logic from streamlined_validate_module.py.
    Returns a tuple of (validated_module, validation_report).
    
    The validated_module will contain only problems that passed validation.
    If all problems fail validation, returns (None, validation_report).
    If the module has no problems, it is considered valid and returned as is.
    
    Uses a single LLM API call to validate all problems in the module.
    """
    import time
    start_time = time.time()
    logger.info(f"Validating module: {teaching_module.module_id}")
    
    # Extract source_utterance_info, ensure it's a dict for the LLM prompt
    module_source_info = teaching_module.source_utterance_info.model_dump() if teaching_module.source_utterance_info else {}
    
    problem_validation_results: List[ProblemValidationResult] = []
    valid_problems = []
    invalid_problems = []
    has_valid_problems = False
    
    if not teaching_module.problems:
        logger.info(f"Module {teaching_module.module_id} has no problems to validate.")
        # For modules without problems (e.g., vocabulary modules), consider them valid
        has_valid_problems = True
        # Return the module as is
        validated_module = teaching_module
    else:
        # Convert all problems to dicts for the LLM prompt
        problem_dicts = [problem.model_dump() for problem in teaching_module.problems]
        problems_by_id = {problem.problem_id: problem for problem in teaching_module.problems}
        num_problems = len(problem_dicts)
        
        # Call the batch validation function with all problems
        validation_start = time.time()
        validation_results = await call_llm_for_batch_problem_validation(module_source_info, problem_dicts)
        validation_end = time.time()
        validation_time = validation_end - validation_start
        
        logger.info(f"Batch validation of {num_problems} problems took {validation_time:.2f} seconds ({validation_time/num_problems:.2f} seconds per problem)")
        
        if validation_results:
            problem_validation_results = validation_results
            valid_count = 0
            invalid_count = 0
            
            # Process each validation result
            for result in validation_results:
                problem_id = result.problem_id
                if problem_id in problems_by_id:
                    if result.overall_problem_validation_passes:
                        # Problem passed validation - keep it
                        valid_problems.append(problems_by_id[problem_id])
                        valid_count += 1
                        has_valid_problems = True
                        logger.info(f"Problem {problem_id} PASSED validation.")
                    else:
                        # Problem failed validation - drop it
                        invalid_problems.append(problems_by_id[problem_id])
                        invalid_count += 1
                        logger.warning(f"Problem {problem_id} FAILED validation - will be dropped.")
                else:
                    logger.warning(f"Validation result for unknown problem ID: {problem_id}")
            
            logger.info(f"Validation summary for {teaching_module.module_id}: {valid_count}/{num_problems} problems passed validation.")
            
            # Create a new module with only the valid problems
            if has_valid_problems:
                validated_module = teaching_module.model_copy()
                validated_module.problems = valid_problems
            else:
                validated_module = None
        else:
            logger.error(f"Failed to get batch validation results for module {teaching_module.module_id}")
            validated_module = None  # Consider a failure to validate as a module with no valid problems
    
    end_time = time.time()
    logger.info(f"Total validation time for module {teaching_module.module_id}: {end_time - start_time:.2f} seconds")
    
    # Create validation report
    report = ModuleValidationReport(
        module_id=teaching_module.module_id,
        overall_module_validation_status=has_valid_problems,  # Module passes if it has at least one valid problem
        problem_validation_results=problem_validation_results
    )
    
    return validated_module, report

async def process_single_recommendation(
    input_rec: InputRecommendation,
    prompts: Dict[str, str],
    module_id_prefix: str,
    default_module_type: str
) -> Tuple[Optional[TeachingModule], Optional[ModuleValidationReport]]:
    processing_start_time = time.time()
    logger.info(
        f"[TIMING] START process_single_recommendation for utterance (original: '{input_rec.original[:30]}...', "
        f"source_file: {input_rec.source_file_type}, utterance_idx: {input_rec.utterance_index_in_file}) "
        f"at {processing_start_time:.2f}"
    )

    # Initialize explanation_parts and problems_for_this_module
    explanation_parts = []
    problems_for_this_module = []

    logger.info(f"Processing recommendation: {input_rec.original[:50]}...")

    # 1. Generate Explanation and Problems (potentially concurrently for Korean phrase utterances)
    explanation = None
    problems: List[Problem] = []
    
    # Determine which explanation prompt to use based on source_file_type
    explanation_prompt_key = "korean_explanation" if input_rec.source_file_type in ["korean_word", "korean_phrase"] else "explanation"
    
    # Determine problem types based on source_file_type
    task_specs = []
    if input_rec.source_file_type == "grammar":
        task_specs = [("mcq", 2), ("fill_blank", 2)]
    elif input_rec.source_file_type == "patterns":
        task_specs = [("dictation", 5)]
    elif input_rec.source_file_type == "korean_phrase":
        # Korean phrases get dictation problems (same number as pattern modules)
        task_specs = [("dictation", 5)]
    # For vocab, korean_word, or other types, no problem tasks
    
    # For Korean phrase utterances, run explanation and problem generation concurrently
    if input_rec.source_file_type == "korean_phrase" and task_specs:
        # Create tasks for both explanation and problems
        explanation_task = asyncio.create_task(generate_explanation(input_rec, prompts[explanation_prompt_key]))
        problem_tasks = [asyncio.create_task(generate_problems(input_rec, ptype, count, prompts)) for ptype, count in task_specs]
        
        # Run all tasks concurrently
        all_results = await asyncio.gather(explanation_task, *problem_tasks)
        
        # Extract results
        explanation = all_results[0]  # First result is the explanation
        for plist in all_results[1:]:  # Remaining results are problem lists
            problems.extend(plist)
    else:
        # For other utterance types, generate explanation first, then problems sequentially
        explanation = await generate_explanation(input_rec, prompts[explanation_prompt_key])
        if explanation and task_specs:
            # Only generate problems if explanation was successful
            tasks = [asyncio.create_task(generate_problems(input_rec, ptype, count, prompts)) for ptype, count in task_specs]
            results = await asyncio.gather(*tasks)
            for plist in results:
                problems.extend(plist)
    
    # Check if explanation generation was successful
    if not explanation:
        logger.error(f"Failed to generate explanation for {input_rec.original[:50]}. Skipping module.")
        return None, None

    if not problems:
        logger.warning(f"No problems generated for {input_rec.original[:50]}. Module will have empty problems list.")

    # 3. Assemble Teaching Module
    module_id = f"module_{module_id_prefix}_{input_rec.utterance_index_in_file:04d}_{uuid.uuid4().hex[:6]}"
    # Determine module_type (can be based on focus_type, priority, etc.)
    module_type = default_module_type # Simplified for now
    if input_rec.focus_type == "PATTERNS": # From the actual JSON value
        module_type = "DRILL_PRACTICE"
    elif input_rec.focus_type == "VOCABULARY":
        module_type = "CONVERSATIONAL_LOW"
    # Add more logic for module_type if needed

    help_context_parts = []
    if input_rec.selected_error:
        help_context_parts.append(f"Focuses on {input_rec.selected_error.type}: {input_rec.selected_error.description}.")
    if input_rec.reason:
        help_context_parts.append(f"Reasoning: {input_rec.reason}")
    help_context = " ".join(help_context_parts) if help_context_parts else None

    teaching_module_shell_if_all_failed = TeachingModule(
        module_id=module_id,
        module_type=module_type,
        source_utterance_info=input_rec, # Pass the rich InputRecommendation object
        explanations=explanation, # This should be LLMExplanation, not LLMExplanationOutput
        problems=problems,
        help_context=help_context
    )

    try:
        # Validate the module before saving
        # The validated_module will contain only problems that passed validation
        validated_module, validation_report = await validate_teaching_module(teaching_module_shell_if_all_failed)
        
        # Save validation report
        report_path = DEFAULT_OUTPUT_DIR / "validation_reports" / f"{teaching_module_shell_if_all_failed.module_id}_validation_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(validation_report.model_dump_json(indent=2, exclude_none=True) if hasattr(validation_report, 'model_dump_json') else validation_report.json(indent=2, exclude_none=True))
            logger.info(f"Saved validation report: {report_path}")
        except Exception as e_report:
            logger.error(f"Error saving validation report for {teaching_module_shell_if_all_failed.module_id}: {e_report}")
        
        # If validated_module is not None, it means at least one problem passed validation
        if validated_module is not None:
            # Save the validated module with only valid problems
            output_file_path = DEFAULT_OUTPUT_DIR / "validated" / f"{teaching_module_shell_if_all_failed.module_id}.json"
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(validated_module.model_dump_json(indent=2) if hasattr(validated_module, 'model_dump_json') else validated_module.json(indent=2))
                
                # Log how many problems were kept vs. total
                original_problem_count = len(teaching_module_shell_if_all_failed.problems) if teaching_module_shell_if_all_failed.problems else 0
                valid_problem_count = len(validated_module.problems) if validated_module.problems else 0
                
                if original_problem_count > 0 and valid_problem_count < original_problem_count:
                    logger.info(f"Module PARTIALLY PASSED validation. Kept {valid_problem_count}/{original_problem_count} problems. Saved to: {output_file_path}")
                else:
                    logger.info(f"Module FULLY PASSED validation. Saved to: {output_file_path}")
                    
                processing_end_time = time.time()
                duration = processing_end_time - processing_start_time
                logger.info(
                    f"[TIMING] END process_single_recommendation for utterance (original: '{input_rec.original[:30]}...', "
                    f"source_file: {input_rec.source_file_type}, utterance_idx: {input_rec.utterance_index_in_file}) "
                    f"at {processing_end_time:.2f}. DURATION: {duration:.2f}s"
                )
                return validated_module, validation_report
            except Exception as e_save:
                logger.error(f"Error saving validated teaching module {teaching_module_shell_if_all_failed.module_id} to JSON: {e_save}")
        else:
            # All problems failed validation - save original module to rejected directory
            output_file_path = DEFAULT_OUTPUT_DIR / "rejected" / f"{teaching_module_shell_if_all_failed.module_id}.json"
            try:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(teaching_module_shell_if_all_failed.model_dump_json(indent=2) if hasattr(teaching_module_shell_if_all_failed, 'model_dump_json') else teaching_module_shell_if_all_failed.json(indent=2))
                logger.warning(f"Module FAILED validation (all problems invalid). Saved to: {output_file_path}")
            except Exception as e_save:
                logger.error(f"Error saving rejected teaching module {teaching_module_shell_if_all_failed.module_id} to JSON: {e_save}")
            processing_end_time = time.time()
            duration = processing_end_time - processing_start_time
            logger.info(
                f"[TIMING] END process_single_recommendation for utterance (original: '{input_rec.original[:30]}...', "
                f"source_file: {input_rec.source_file_type}, utterance_idx: {input_rec.utterance_index_in_file}) "
                f"ERROR_CASE at {processing_end_time:.2f}. DURATION: {duration:.2f}s"
            )
            return None, validation_report

    except Exception as e:
        logger.error(f"Error processing recommendation {input_rec.original[:50]}: {e}")
        # Ensure a ValidationReport is created even in case of early exit due to error
        error_report = ModuleValidationReport(
            module_id=f"{module_id_prefix}_{input_rec.utterance_index_in_file}_ERROR",
            overall_validation_status="ERROR_IN_GENERATION",
            failure_reason=f"Unhandled exception: {str(e)}",
            problem_reports=[]
        )
        processing_end_time = time.time()
        duration = processing_end_time - processing_start_time
        logger.info(
            f"[TIMING] END process_single_recommendation for utterance (original: '{input_rec.original[:30]}...', "
            f"source_file: {input_rec.source_file_type}, utterance_idx: {input_rec.utterance_index_in_file}) "
            f"ERROR_CASE at {processing_end_time:.2f}. DURATION: {duration:.2f}s"
        )
        return None, error_report

async def main(grammar_input_path=None, patterns_input_path=None, vocab_small_grammar_input_path=None, output_directory_path=None, korean_word_input_path=None, korean_phrase_input_path=None):
    """Main function to orchestrate teaching module generation.
    
    Args:
        grammar_input_path: Path to the grammar input file
        patterns_input_path: Path to the patterns input file
        vocab_small_grammar_input_path: Path to the vocabulary and small grammar input file
        output_directory_path: Path to the output directory
        korean_word_input_path: Path to the Korean word focused input file
        korean_phrase_input_path: Path to the Korean phrase focused input file
    """
    # Set up input and output paths
    input_paths = {}
    if grammar_input_path:
        input_paths["grammar"] = Path(grammar_input_path)
    if patterns_input_path:
        input_paths["patterns"] = Path(patterns_input_path)
    if vocab_small_grammar_input_path:
        input_paths["vocab"] = Path(vocab_small_grammar_input_path)
    # Only include Korean word and phrase files if they exist and have content
    if korean_word_input_path and Path(korean_word_input_path).exists():
        # Check if the file has actual content (not just an empty structure)
        try:
            with open(korean_word_input_path, 'r', encoding='utf-8') as f:
                korean_word_data = json.load(f)
                if isinstance(korean_word_data, dict) and korean_word_data.get('analysis_zones') and any(zone.get('recommendations') for zone in korean_word_data.get('analysis_zones', [])):
                    input_paths["korean_word"] = Path(korean_word_input_path)
                    logger.info(f"Including Korean word focused file: {korean_word_input_path}")
                else:
                    logger.info(f"Skipping empty Korean word focused file: {korean_word_input_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading Korean word focused file: {e}")
    
    if korean_phrase_input_path and Path(korean_phrase_input_path).exists():
        # Check if the file has actual content (not just an empty structure)
        try:
            with open(korean_phrase_input_path, 'r', encoding='utf-8') as f:
                korean_phrase_data = json.load(f)
                if isinstance(korean_phrase_data, dict) and korean_phrase_data.get('analysis_zones') and any(zone.get('recommendations') for zone in korean_phrase_data.get('analysis_zones', [])):
                    input_paths["korean_phrase"] = Path(korean_phrase_input_path)
                    logger.info(f"Including Korean phrase focused file: {korean_phrase_input_path}")
                else:
                    logger.info(f"Skipping empty Korean phrase focused file: {korean_phrase_input_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading Korean phrase focused file: {e}")
    
    # If no input paths provided, use defaults
    if not input_paths:
        for source_key, config in INPUT_FILES_CONFIG.items():
            input_paths[source_key] = DEFAULT_INPUT_DIR / config["file_name"]
    
    # Set up output directory
    output_dir = Path(output_directory_path) if output_directory_path else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validated").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_reports").mkdir(parents=True, exist_ok=True)
    
    total_modules_generated = 0

    # Load all prompt templates once
    try:
        prompts = {name: load_prompt_template(name) for name in PROMPT_PATHS.keys()}
    except FileNotFoundError:
        logger.critical("Essential prompt template(s) missing. Exiting.")
        return

    for source_key, input_file_path in input_paths.items():
        if not input_file_path.exists():
            logger.warning(f"Input file {input_file_path} does not exist. Skipping.")
            continue
            
        logger.info(f"--- Processing input file: {input_file_path} ---")
        
        recommendations = load_and_parse_input_file(input_file_path, source_key)
        if not recommendations:
            logger.warning(f"No recommendations loaded from {input_file_path}. Skipping.")
            continue

        file_modules_generated = 0

        for i, rec in enumerate(recommendations): # Process recommendations sequentially
            rec.utterance_index_in_file = i 
            rec.source_file_type = source_key

            try:
                # Get the configuration for this source_key
                source_config = INPUT_FILES_CONFIG.get(source_key, {
                    "module_id_prefix": source_key,
                    "default_module_type": "CONVERSATIONAL_LOW"
                })
                
                # Await each recommendation processing call directly
                teaching_module_result, validation_report = await process_single_recommendation(
                    rec,
                    prompts,
                    source_config["module_id_prefix"],
                    source_config["default_module_type"]
                )

                if isinstance(teaching_module_result, TeachingModule):
                    file_modules_generated += 1
                elif teaching_module_result is None:
                     logger.warning(f"Skipped saving module for recommendation {i} from {input_file_path} as processing returned None or failed silently.")

            except KeyError as ke_main_loop:
                if 'reason' in str(ke_main_loop).lower():
                    logger.error(f"KeyError ('reason') caught in main loop for item {i} from {input_file_path}. Offending rec: {json.dumps(rec.model_dump() if isinstance(rec, BaseModel) else rec, indent=2)}", exc_info=True)
                else:
                    logger.error(f"KeyError caught in main loop for item {i} from {input_file_path}: {repr(ke_main_loop)}. Offending rec: {json.dumps(rec.model_dump() if isinstance(rec, BaseModel) else rec, indent=2)}", exc_info=True)
            except Exception as e: # This is the block that seems to be catching the error
                logger.error(f"General error processing recommendation {i} from {input_file_path}: {e}", exc_info=True)
        
        logger.info(f"Generated {file_modules_generated} modules from {input_file_path}.")
        total_modules_generated += file_modules_generated

    logger.info(f"--- Teaching module generation complete. ---")
    logger.info(f"Total modules generated: {total_modules_generated}")
    logger.info(f"Validated modules directory: {(output_dir / 'validated').resolve()}")
    logger.info(f"Rejected modules directory: {(output_dir / 'rejected').resolve()}")
    logger.info(f"Validation reports directory: {(output_dir / 'validation_reports').resolve()}")

    return total_modules_generated

if __name__ == "__main__":
    asyncio.run(main())
