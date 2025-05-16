# Placeholder for Preprocessing Logic
import logging
import os
import json
from typing import List, Dict, Any, Tuple, Optional, get_args, get_origin, Literal, Union
from analyzer_service.schemas import AlignedClause, PreprocessedASUnit
from pathlib import Path

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage 
from dotenv import load_dotenv

# Schema imports
from .schemas import ( 
    InputUtterance,
    TranslatedUtterance,
    SegmentedASUnit,
    AlignedASUnit,
    Clause, 
    AlignedClause, 
    PreprocessingOutput
) 

# Schema for input
# Placeholder for the eventual analysis results schema
AnalysisResult = Dict 

# Set up logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PROMPT_DIR = Path(__file__).parent / "prompts"
TRANSLATION_PROMPT_FILE = PROMPT_DIR / "translation_prompt.txt"
SEGMENTATION_PROMPT_FILE = PROMPT_DIR / "segmentation_prompt.txt"
ALIGNMENT_PROMPT_FILE = PROMPT_DIR / "as_unit_alignment_prompt.txt"
CLAUSE_IDENTIFICATION_PROMPT_FILE = PROMPT_DIR / "clause_identification_prompt.txt"
# Consider making the model name configurable
MODEL_NAME = "gemini-2.0-flash"

# --- Helper Functions ---
def count_words(text: str) -> int:
    """Counts the number of words in a string."""
    if not text:
        return 0
    # Simple split by whitespace, might need refinement for complex cases
    return len(text.split())

def load_and_parse_prompt(file_path: str) -> tuple[str, str]:
    """Loads prompt text and splits into system instruction and human template."""
    logger.debug(f"Loading prompt from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine split marker and variable based on file name convention
        split_marker = "## HUMAN INSTRUCTION:"
        human_var = "{text}" # Default
        output_suffix = "" # Default

        if "translation" in os.path.basename(file_path):
            split_marker = "## INPUT:" # CORRECTED: Use the actual marker from the file
            human_var = "{korean_english_text}" # Use the variable name from the prompt file
            output_suffix = "\n\n## OUTPUT:\nTranslation:" # Match output section
        elif "segmentation" in os.path.basename(file_path):
            split_marker = "## Input Utterance" # CORRECTED: Actual marker from file
            human_var = "{input_utterances}" # CORRECTED: Actual variable from file
            output_suffix = "\n\n## Required Output Format" # Match output section
        elif "as_unit_alignment" in os.path.basename(file_path):
            split_marker = "## Input" # CORRECTED: Actual marker from file
            # Construct the human variable part dynamically for multiple inputs
            # Note: LangChain's template syntax uses single curly braces
            human_var_content = "Original Utterance Text:\n{original_input_text}\n\nTarget AS Unit Text:\n{target_as_unit_text}"
            output_suffix = "\n\n## Output" # Match output section
        elif "clause_identification" in os.path.basename(file_path):
            split_marker = "## INPUT:" # CORRECTED: Actual marker from file
            human_var = "{as_unit_text}" # Use the variable name from the prompt file
            output_suffix = "\n\n## OUTPUT:" # Match output section
        else:
            # Default or fallback split logic if needed
            logger.warning(f"Unknown prompt type for parsing: {file_path}. Using default split.")

        # Split the content
        parts = content.split(split_marker, 1)
        if len(parts) == 2:
            system_instruction = parts[0].strip()
            # Reconstruct the human part including the marker and variable
            if "as_unit_alignment" in os.path.basename(file_path):
                human_template = f"{split_marker}\n{human_var_content}\n{output_suffix}".strip()
            else:
                # Use standard f-string interpolation for single variables
                human_template = f"{split_marker}\n{{{human_var.strip('{}')}}}\n{output_suffix}".strip()
            return system_instruction, human_template
        else:
            logger.error(f"Could not find '{split_marker}' in prompt file: {file_path}")
            raise ValueError(f"Prompt file '{file_path}' is not structured correctly.")

    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading or parsing prompt file {file_path}: {e}")
        raise

def parse_segmentation_output(raw_text: str) -> List[str]:
    """Robustly parses LLM segmentation output, handling various formats."""
    cleaned_text = raw_text.strip()
    # Remove common markdown fences
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    as_units = []
    known_text_keys = ["as_unit", "segment_text", "text"]

    # 1. Try parsing as JSON
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    as_units.append(item.strip())
                elif isinstance(item, dict):
                    for key in known_text_keys:
                        if key in item and isinstance(item[key], str):
                            segment = item[key].strip()
                            if segment:
                                as_units.append(segment)
                            break # Found text in this dict item
            # If we successfully parsed JSON and it was a list (even if empty),
            # return the result derived from JSON parsing.
            if as_units:
                logger.debug(f"Successfully parsed segmentation output as JSON list. Found {len(as_units)} units.")
            else:
                logger.debug("Successfully parsed segmentation output as JSON, resulting in an empty list of units.")
            # Apply final cleaning to units derived from JSON
            cleaned_units = []
            for unit in as_units:
                # Remove "segment": prefix if present
                segment_prefix = '"segment":'
                utterance_prefix = 'utterance\":'
                cleaned_unit = unit # Start with original unit
                if cleaned_unit.lower().startswith(segment_prefix):
                    cleaned_unit = cleaned_unit[len(segment_prefix):].strip()
                elif cleaned_unit.lower().startswith(utterance_prefix):
                    cleaned_unit = cleaned_unit[len(utterance_prefix):].strip()
                # Replace escaped quotes
                cleaned_unit = cleaned_unit.replace('\\"', '"')
                # Explicitly remove leading/trailing quotes
                if cleaned_unit.startswith('"'):
                    cleaned_unit = cleaned_unit[1:]
                if cleaned_unit.endswith('"'):
                    cleaned_unit = cleaned_unit[:-1]
                # Remove trailing comma and final strip
                cleaned_unit = cleaned_unit.strip().strip(',')
                cleaned_units.append(cleaned_unit.strip())
            return [u for u in cleaned_units if u] # Filter empty strings again
        else:
             logger.warning(f"Parsed JSON but it was not a list: {type(data)}")

    except json.JSONDecodeError:
        logger.debug("Segmentation output was not valid JSON. Falling back to line parsing.")
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing of segmentation output: {e}. Raw text:\n{raw_text}")

    # 2. Fallback to line-by-line parsing (if JSON failed or yielded nothing)
    logger.debug("Applying line-by-line parsing fallback for segmentation.")
    lines = raw_text.split('\n')
    potential_units = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, fences, list markers, or lines looking like JSON structures
        if not line or \
           line in ('[', ']', '{', '}', '-') or \
           line.startswith(('```', '*', '#', '1.', '2.', 'AS-unit')):
            continue
        # Handle markdown list items
        if line.startswith('- '):
            line = line[2:].strip()
        # Basic check to avoid adding partial JSON lines
        if '"as_unit"' in line or '"segment_text"' in line or '"implied_elements"' in line:
            continue
        # CLEANING STEP ADDED HERE for fallback
        # Remove "segment": prefix if present
        segment_prefix = '"segment":'
        utterance_prefix = 'utterance\":'
        cleaned_line = line # Start with original line
        if cleaned_line.lower().startswith(segment_prefix):
            cleaned_line = cleaned_line[len(segment_prefix):].strip()
        elif cleaned_line.lower().startswith(utterance_prefix):
            cleaned_line = cleaned_line[len(utterance_prefix):].strip()
        # Replace escaped quotes
        cleaned_line = cleaned_line.replace('\\"', '"')
        # Explicitly remove leading/trailing quotes
        if cleaned_line.startswith('"'):
            cleaned_line = cleaned_line[1:]
        if cleaned_line.endswith('"'):
            cleaned_line = cleaned_line[:-1]
        # Remove trailing comma and final strip
        cleaned_line = cleaned_line.strip().strip(',')
        if cleaned_line:
            as_units.append(cleaned_line.strip())
    logger.debug(f"Segmentation fallback parsing produced: {as_units}")

    # Final filter for empty strings just in case
    as_units = [unit for unit in as_units if unit]

    if not as_units:
        logger.warning(f"Could not extract any segmented units from raw output:\n{raw_text}")
        return []
    return as_units

def parse_clause_identification_output(raw_output: str) -> List[str]:
    """Parses the potentially messy output of the clause identification LLM call.

    Attempts to parse as a JSON list of objects, where each object contains:
        - clause_text (str)

    Returns the list of clause text strings or an empty list if parsing/validation fails.
    """
    clause_texts = []
    cleaned_output = raw_output.strip()
    logger = logging.getLogger(__name__)

    # Remove potential markdown fences (```json ... ``` or ``` ... ```)
    if cleaned_output.startswith("```json") and cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[7:-3].strip()
    elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[3:-3].strip()

    # Attempt JSON parsing and validation
    try:
        parsed_json = json.loads(cleaned_output)
        if not isinstance(parsed_json, list):
            logger.warning(f"Clause identification JSON parsed, but not a list: {type(parsed_json)}")
            return []

        # Extract clause_text from each item in the list
        validated_texts = []
        for i, item in enumerate(parsed_json):
            if not isinstance(item, dict):
                logger.warning(f"Clause identification item {i} is not a dictionary: {type(item)}. Skipping.")
                continue # Skip invalid item

            clause_text = item.get('clause_text')

            if not isinstance(clause_text, str) or not clause_text.strip():
                logger.warning(f"Clause identification item {i} missing or invalid 'clause_text'. Skipping.")
                continue

            validated_texts.append(clause_text.strip())

        if validated_texts:
             logger.debug(f"Clause identification JSON parsing successful for {len(validated_texts)} clauses.")
             return validated_texts
        else:
            logger.warning("Clause identification JSON parsed as list, but no valid clause_text found.")
            return []

    except json.JSONDecodeError as e:
        logger.warning(f"Clause identification JSON parsing failed: {e}. Raw output: '{cleaned_output[:100]}...'", exc_info=True)
        return [] # Return empty list on JSON failure
    except Exception as e:
        logger.error(f"Unexpected error during clause identification JSON parsing: {e}. Raw output: '{cleaned_output[:100]}...'", exc_info=True)
        return [] # Return empty list on other unexpected errors

def parse_clause_alignment_output(raw_output: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Parses the potentially string or already-parsed output of the clause alignment LLM call.

    Args:
-        raw_output: The raw string output from the LLM.
+        raw_output: The raw string output from the LLM or an already parsed list.

    Returns:
        A list of dictionaries, where each dictionary represents an aligned clause
    """
    if isinstance(raw_output, str):
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json") and cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[7:-3].strip()
        elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[3:-3].strip()

        try:
            parsed_list = json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid alignment output type: {type(raw_output)}. Expected str or list.", exc_info=True)
            return []
    elif isinstance(raw_output, list):
        parsed_list = raw_output
    else:
        logger.error(f"Invalid alignment output type: {type(raw_output)}. Expected str or list.", exc_info=True)
        return []

    if not isinstance(parsed_list, list):
        logger.error(f"Alignment output is not a list: {parsed_list}")
        return []

    validated_alignments = []
    for item in parsed_list:
        # Validate that each item in the list is a dictionary
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dictionary item in alignment list: {item}")
            continue

        validated_alignments.append(item)

    return validated_alignments

# --- Chain Initialization ---
def initialize_translation_chain():
    """Initializes the translation chain."""
    load_dotenv() # Load environment variables (like GOOGLE_API_KEY)

    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY must be set.")

    try:
        system_instruction, human_template = load_and_parse_prompt(TRANSLATION_PROMPT_FILE)

        translation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_instruction),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, convert_system_message_to_human=True) # Use configured model
        output_parser = StrOutputParser()

        chain = (
            translation_prompt
            | llm
            | output_parser
        )
        logger.info("Translation chain initialized successfully.")
        return chain

    except Exception as e:
        logger.error(f"Failed to initialize translation chain: {e}")
        raise

def initialize_segmentation_chain():
    """Initializes the segmentation chain."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY missing for segmentation chain.")
        raise ValueError("GOOGLE_API_KEY must be set.")

    try:
        system_instruction, human_template = load_and_parse_prompt(SEGMENTATION_PROMPT_FILE)

        segmentation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_instruction),
            # Ensure the variable name here matches the key used in batch input later
            HumanMessagePromptTemplate.from_template(human_template) # template expects {translated_text}
        ])

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, convert_system_message_to_human=True) # Use configured model
        # Chain: Prompt -> LLM -> JSON Output -> Parse into List[str]
        chain = (
            segmentation_prompt
            | llm
            | JsonOutputParser() # Use JSON parser
        )
        logger.info("Segmentation chain initialized successfully.")
        return chain

    except Exception as e:
        logger.error(f"Failed to initialize segmentation chain: {e}")
        raise

def initialize_alignment_chain():
    """Initializes the AS Unit alignment chain."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY missing for alignment chain.")
        raise ValueError("GOOGLE_API_KEY must be set.")

    try:
        system_instruction, human_template = load_and_parse_prompt(ALIGNMENT_PROMPT_FILE)

        alignment_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_instruction),
            HumanMessagePromptTemplate.from_template(human_template) # Expects {original_input_text}, {target_as_unit_text}
        ])

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, convert_system_message_to_human=True)

        # Use StrOutputParser to get the raw aligned segment or 'NO_ALIGNMENT_FOUND'
        chain = (
            alignment_prompt
            | llm
            | StrOutputParser()
        )
        logger.info("AS Unit Alignment chain initialized successfully.")
        return chain

    except Exception as e:
        logger.error(f"Failed to initialize AS Unit Alignment chain: {e}")
        raise

def initialize_clause_identification_chain():
    """Initializes the clause identification chain."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY missing for clause identification chain.")
        raise ValueError("GOOGLE_API_KEY must be set.")

    try:
        system_instruction, human_template = load_and_parse_prompt(CLAUSE_IDENTIFICATION_PROMPT_FILE)

        identification_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_instruction),
            HumanMessagePromptTemplate.from_template(human_template) # Expects {as_unit_text}
        ])

        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, convert_system_message_to_human=True)

        # Use StrOutputParser to get the raw identified clauses
        chain = (
            identification_prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Clause Identification chain initialized successfully.")
        return chain

    except Exception as e:
        logger.error(f"Failed to initialize Clause Identification chain: {e}")
        raise

def initialize_clause_alignment_chain():
    """Initializes the LangChain chain for clause alignment and characterization."""
    try:
        # Create explicit system and human messages for more control
        system_message = """You are a language analysis assistant specializing in aligning and characterizing clauses across languages. 
        Your task is to identify how English clauses map to segments in the original text (which may be Korean, English, or mixed) 
        and characterize those original segments.
        
        For each provided English clause, identify its corresponding segment within the original text and characterize that segment.
        
        You MUST output a valid JSON list where each item corresponds to an input clause and contains:
        - clause_id: The exact same clause_id from the input
        - aligned_original_clause_segment: The text segment from the original text that corresponds to the English clause
        - original_clause_type: One of 'word', 'phrase', or 'collocation'
        
        Ensure the output list has the same number of items as the input clauses list and maintains the order.
        Do NOT include any explanatory text or markdown formatting around the JSON output."""
        
        human_template = """### Aligned Original Text for AS Unit:
{aligned_original_text}

### Identified English Clauses (List of Dicts):
{clauses}

## Your Output (JSON list ONLY):"""

        # Create a ChatPromptTemplate with explicit messages
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])

        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, convert_system_message_to_human=True, temperature=0.1)

        # Define the chain
        chain = (
            prompt
            | llm
            | JsonOutputParser()
        )
        logger.info("Clause Alignment chain initialized successfully.")
        return chain
    except Exception as e:
        logger.error(f"Error initializing clause alignment chain: {e}")
        raise

# --- Main Batch Processing Function ---
# Initialize chains once when the module is loaded
translation_chain = initialize_translation_chain()
segmentation_chain = initialize_segmentation_chain()
alignment_chain = initialize_alignment_chain()
clause_identification_chain = initialize_clause_identification_chain()
clause_alignment_chain = initialize_clause_alignment_chain()

# Update output type hint if using Pydantic models later
# from .schemas import SegmentedASUnit
def run_preprocessing_batch(batch: List[InputUtterance]) -> List[PreprocessedASUnit]: 
    """Processes a batch through translation, filtering, and segmentation."""
    logger.info(f"[Preprocessing Module] Received batch of {len(batch)} utterances.")

    if not batch:
        logger.warning("[Preprocessing Module] Received empty batch.")
        return []

    # Prepare input for translation
    batch_input_translation = [{"korean_english_text": item.text} for item in batch] 

    try:
        # 1. --- Translation Step ---
        logger.info(f"Invoking translation chain batch with {len(batch_input_translation)} items...")
        translations = translation_chain.batch(batch_input_translation)
        logger.info("Translation chain batch call completed.")

        translated_batch_items = []
        for i, item in enumerate(batch): # Iterate through original InputUtterance objects
            translation_text = translations[i]
            # Clean potential prefix if LLM adds it (though prompt attempts to prevent this)
            prefix_to_remove = "Translation:"
            if translation_text.strip().startswith(prefix_to_remove):
                translation_text = translation_text.strip()[len(prefix_to_remove):].strip()

            translated_batch_items.append({
                "utterance_id": item.id, 
                "original_input_text": item.text, 
                "translated_text": translation_text,
            })
        logger.info(f"Finished translation. {len(translated_batch_items)} items resulted.")

        # 2. --- Filtering Step ---
        min_words = 3
        filtered_batch = [item for item in translated_batch_items if count_words(item.get('translated_text', '')) >= min_words]
        items_filtered_out = len(translated_batch_items) - len(filtered_batch)
        if items_filtered_out > 0:
            logger.info(f"Filtered out {items_filtered_out} items with < {min_words} words.")
        logger.info(f"Batch size after filtering: {len(filtered_batch)}")

        if not filtered_batch:
             logger.warning("[Preprocessing Module] Batch empty after filtering.")
             return []

        # 3. --- Segmentation Step ---
        # Prepare input for segmentation chain - key must match prompt template variable
        batch_input_segmentation = [{"input_utterances": item['translated_text']} for item in filtered_batch] 
        logger.info(f"Invoking segmentation chain batch with {len(batch_input_segmentation)} items...")
        segmentation_raw_results = segmentation_chain.batch(batch_input_segmentation)
        logger.info("Segmentation chain batch call completed.")

        if len(segmentation_raw_results) != len(filtered_batch):
            logger.error(f"Segmentation mismatch: {len(filtered_batch)} vs {len(segmentation_raw_results)}")
            # Decide how to handle: return partial, return error, etc.
            # For now, let's log and continue, potentially leaving some units unsegmented
            pass # Allow processing to continue, but log the error

        # Process the segmentation results (already parsed by JsonOutputParser)
        parsed_segmentation_results: List[List[str]] = []
        for raw_result in segmentation_raw_results:
            segments_for_utterance = []
            if isinstance(raw_result, list):
                # Expecting list of strings or list of dicts
                for item in raw_result:
                    segment_text = None
                    if isinstance(item, str):
                        segment_text = item.strip()
                    elif isinstance(item, dict):
                        # Try common keys used in segmentation prompts
                        for key in ["segment", "as_unit", "utterance", "text", "segment_text"]:
                            if key in item and isinstance(item[key], str):
                                segment_text = item[key].strip()
                                break # Found text in this dict item
                    if segment_text:
                        segments_for_utterance.append(segment_text)
                    else:
                         logger.warning(f"Could not extract segment string from item: {item} in raw_result: {raw_result}")
            elif isinstance(raw_result, dict):
                # Sometimes the parser might return a dict with a key containing the list
                found_list = None
                for key in ["segments", "as_units", "result", "output"]:
                    if key in raw_result and isinstance(raw_result[key], list):
                        found_list = raw_result[key]
                        break
                if found_list:
                    logger.debug(f"Found segment list under key '{key}' in dict result.")
                     # Re-process the found list (similar logic as above)
                    for item in found_list:
                        segment_text = None
                        if isinstance(item, str):
                            segment_text = item.strip()
                        elif isinstance(item, dict):
                             for key2 in ["segment", "as_unit", "utterance", "text", "segment_text"]:
                                if key2 in item and isinstance(item[key2], str):
                                    segment_text = item[key2].strip()
                                    break
                        if segment_text:
                            segments_for_utterance.append(segment_text)
                        else:
                            logger.warning(f"Could not extract segment string from item: {item} in nested list.")
                else:
                    logger.warning(f"Segmentation result was a dict, but couldn't find list key: {raw_result}")
            else:
                logger.warning(f"Unexpected segmentation result format: {type(raw_result)}. Expected list or dict. Value: {raw_result}")

            parsed_segmentation_results.append(segments_for_utterance)

        # Ensure result length matches filtered batch length
        if len(parsed_segmentation_results) != len(filtered_batch):
            logger.error(f"Segmentation result length mismatch: {len(filtered_batch)} vs {len(parsed_segmentation_results)}")

        # Flatten results into list of AS Units matching SegmentedASUnit structure (as dicts)
        all_segmented_units = []
        # Iterate using the *parsed* results
        for original_item, as_unit_texts in zip(filtered_batch, parsed_segmentation_results):
            if not as_unit_texts: # Handle cases where parsing yielded no units
                 logger.warning(f"Parsing yielded no units for utterance {original_item['utterance_id']}. Check raw LLM output if possible.")
                 # logger.debug(f"Original raw LLM output for {original_item['utterance_id']}:\n{segmentation_raw_results[filtered_batch.index(original_item)]}") # Optional: log raw text again if parse fails
                 continue # Skip this utterance
            for i, as_unit_text in enumerate(as_unit_texts):
                 as_unit_id = f"{original_item['utterance_id']}-as{i+1}"
                 # Corresponds to SegmentedASUnit schema fields (approximated as dict)
                 segmented_unit = {
                     "original_utterance_id": original_item['utterance_id'],
                     "as_unit_id": as_unit_id,
                     "original_input_text": original_item['original_input_text'], 
                     "as_unit_text": as_unit_text, 
                     # Add placeholder fields that might be populated later (or use Pydantic default)
                     "aligned_original_text": None, 
                     "status": "segmented", 
                     "clauses": [], 
                     "complexity_score": None,
                     "accuracy_score": None,
                     "errors_found": [],
                     "pattern_analysis": {},
                     "teaching_content": None
                 }
                 all_segmented_units.append(segmented_unit)

        logger.info(f"[Preprocessing Module] Finished segmentation. Produced {len(all_segmented_units)} AS units.")

        # 4. --- AS Unit Alignment Step ---
        if not all_segmented_units:
            logger.info("[Preprocessing Module] No AS units to align. Skipping alignment step.")
            return [] # Nothing left to process

        # Prepare batch input for alignment
        batch_input_alignment = [
            {
                "original_input_text": unit["original_input_text"],
                "target_as_unit_text": unit["as_unit_text"]
            }
            for unit in all_segmented_units
        ]
        logger.info(f"Invoking alignment chain batch with {len(batch_input_alignment)} AS units...")
        alignment_raw_results = alignment_chain.batch(batch_input_alignment)
        logger.info("Alignment chain batch call completed.")

        if len(alignment_raw_results) != len(all_segmented_units):
            logger.error(f"Alignment mismatch: {len(all_segmented_units)} units vs {len(alignment_raw_results)} results")
            # Decide how to handle: return partial, return error, etc.
            # For now, let's log and continue, potentially leaving some units unaligned
            pass # Allow processing to continue, but log the error

        # Add aligned text to the units
        for i, unit in enumerate(all_segmented_units):
            if i < len(alignment_raw_results):
                raw_aligned_text = alignment_raw_results[i].strip()
                # Handle the specific 'NO_ALIGNMENT_FOUND' case
                if raw_aligned_text == "NO_ALIGNMENT_FOUND":
                    unit["aligned_original_text"] = None
                    unit["status"] = "alignment_failed"
                    logger.debug(f"Alignment failed for AS Unit {unit['as_unit_id']}")
                else:
                    unit["aligned_original_text"] = raw_aligned_text
                    unit["status"] = "aligned"
                    logger.debug(f"Alignment successful for AS Unit {unit['as_unit_id']}")
            else:
                 # This happens if alignment_raw_results was shorter than all_segmented_units
                 unit["aligned_original_text"] = None
                 unit["status"] = "alignment_skipped_error"
                 logger.warning(f"Skipping alignment for AS Unit {unit['as_unit_id']} due to result mismatch.")

        logger.info(f"[Preprocessing Module] Finished AS Unit alignment.")

        # 5. --- Clause Identification Step ---
        logger.info(f"Invoking clause identification chain batch with {len(all_segmented_units)} AS units...")
        clause_identification_batch_input = [
            {"as_unit_text": unit['as_unit_text']} for unit in all_segmented_units
        ]
        clause_identification_raw_results = clause_identification_chain.batch(clause_identification_batch_input, config={"max_concurrency": 5})
        logger.info("Clause identification chain batch call completed.")

        # Process clause identification results
        for i, unit in enumerate(all_segmented_units):
            if i < len(clause_identification_raw_results):
                raw_identified_clauses = clause_identification_raw_results[i]
                # Get List[str] from parser
                clause_text_list: List[str] = parse_clause_identification_output(raw_identified_clauses)
                
                # 1. Convert strings to Clause objects first
                identified_clauses_base: List[Clause] = []
                for idx, clause_text in enumerate(clause_text_list):
                    if clause_text.strip():  # Skip empty strings
                        clause_id = f"{unit['as_unit_id']}-c{idx}"
                        # Create the base Clause object
                        new_clause = Clause(clause_id=clause_id, clause_text=clause_text.strip())
                        identified_clauses_base.append(new_clause)

                # 2. Convert the list of Clause objects to AlignedClause objects
                aligned_clauses: List[AlignedClause] = [
                    AlignedClause(**c.model_dump()) for c in identified_clauses_base
                ]

                unit['clauses'] = aligned_clauses # Assign the list of AlignedClause
                unit['status'] = 'clause_identified'
                logger.debug(f"Clause identification successful for AS Unit {unit['as_unit_id']}")
            else:
                # This happens if clause_identification_raw_results was shorter than all_segmented_units
                unit['clauses'] = []
                unit['status'] = 'clause_identification_skipped_error'
                logger.warning(f"Skipping clause identification for AS Unit {unit['as_unit_id']} due to result mismatch.")

        logger.info(f"[Preprocessing Module] Finished clause identification.")

        # --- 5. Clause Alignment and Characterization --- 
        # Prepare batch input for clause alignment
        clause_alignment_batch_input = []
        for unit in all_segmented_units:
            # Convert Pydantic Clause models to simple dicts for JSON serialization
            # Only include clause_id and clause_text for the prompt input
            clauses_list_for_prompt = [
                {"clause_id": clause.clause_id, "clause_text": clause.clause_text}
                for clause in unit['clauses']
            ]
            clause_alignment_batch_input.append(
                {
                    "aligned_original_text": unit.get('aligned_original_text', ""), 
                    "clauses": clauses_list_for_prompt # Pass the list directly
                }
            )

        if not clause_alignment_batch_input:
            logger.info("Skipping clause alignment as there are no AS units with clauses.")
            clause_alignment_results = []
        else:
            logger.info(f"Invoking clause alignment chain batch with {len(clause_alignment_batch_input)} items...")
            try:
                # Assuming clause_alignment_chain is initialized globally
                clause_alignment_results = clause_alignment_chain.batch(clause_alignment_batch_input, config={"max_concurrency": 5})
                logger.info("Clause alignment chain batch call completed.")
                # --- DEBUG LOGGING START ---
                # Log the raw structure received from the LLM/parser for each unit's clauses
                logger.debug(f"Raw clause_alignment_results (type: {type(clause_alignment_results)}, len: {len(clause_alignment_results) if isinstance(clause_alignment_results, list) else 'N/A'}):")
                if isinstance(clause_alignment_results, list):
                    for idx, result_item in enumerate(clause_alignment_results):
                        logger.debug(f"  Result item {idx} (type: {type(result_item)}): {json.dumps(result_item, indent=2, ensure_ascii=False)}")
                else:
                    logger.debug(f"  Unexpected type for clause_alignment_results: {clause_alignment_results}")
                # --- DEBUG LOGGING END ---
            except Exception as e:
                logger.error(f"Clause alignment chain batch call failed: {e}", exc_info=True)
                # Handle failure - perhaps create dummy results or re-raise
                # For now, create dummy results with None to match expected structure length
                clause_alignment_results = [None] * len(clause_alignment_batch_input)

        # Process clause alignment results
        if len(all_segmented_units) != len(clause_alignment_results):
            logger.error(f"Mismatch between number of AS units ({len(all_segmented_units)}) and clause alignment results ({len(clause_alignment_results)}). Skipping alignment updates.")
        else:
            for unit, alignment_result in zip(all_segmented_units, clause_alignment_results): 
                # Check if unit is a dict with expected keys
                if not isinstance(unit, dict) or 'as_unit_id' not in unit or 'clauses' not in unit:
                    logger.warning(f"Skipping unit as it does not appear to be a valid AS unit dict: {unit}")
                    continue
                # Further check if 'clauses' is a list
                if not isinstance(unit.get('clauses'), list):
                    logger.warning(f"Skipping unit {unit.get('as_unit_id')} as 'clauses' is not a list: {unit.get('clauses')}")
                    continue
 
                # alignment_result is expected to be the already parsed list due to JsonOutputParser
                if not isinstance(alignment_result, list):
                    logger.warning(f"Expected a list for alignment result for AS Unit {unit['as_unit_id']}, but got {type(alignment_result)}. Skipping.")
                    continue
                    
                # Use the alignment_result directly as the parsed list
                parsed_alignment_list = alignment_result 

                # Create a map from clause_id to alignment data for efficient lookup
                alignment_data_map = {}
                for item in parsed_alignment_list:
                    if isinstance(item, dict) and 'clause_id' in item:
                        alignment_data_map[item['clause_id']] = item
                    else:
                        logger.warning(f"Invalid item format in parsed alignment data for AS Unit {unit['as_unit_id']}: {item}")

                # Optional: Check mismatch between parsed list length and unit['clauses'] length
                # if len(parsed_alignment_list) != len(unit['clauses']):
                #     logger.warning(f"Mismatch in clause count for AS Unit {unit['as_unit_id']}. Expected {len(unit['clauses'])}, got {len(parsed_alignment_list)} alignment results.")

                # Update clauses within the unit
                for clause in unit['clauses']: 
                    if not hasattr(clause, 'clause_id'):
                        logger.warning(f"Skipping clause object as it lacks 'clause_id': {clause}")
                        continue

                    if clause.clause_id in alignment_data_map:
                        data = alignment_data_map[clause.clause_id]
                        assigned_segment = data.get('aligned_original_clause_segment')
                        assigned_type = data.get('original_clause_type')
                        
                        clause.aligned_original_clause_segment = assigned_segment
                        clause.original_clause_type = assigned_type
                        
                        # Correctly implement case-insensitive and whitespace-stripping validation for 'original_clause_type'
                        received_type = data.get('original_clause_type')
                        validated_type = None
                        allowed_literals = set() # Initialize as empty set
                                
                        # Safely extract allowed literals from Optional[Literal[...]]
                        try:
                            field_annotation = AlignedClause.model_fields['original_clause_type'].annotation
                            union_args = get_args(field_annotation) # e.g., (Literal['word',...], type(None))
                            if union_args:
                                for arg in union_args:
                                    if get_origin(arg) is Literal:
                                        literal_args = get_args(arg)
                                        allowed_literals.update(literal_args)
                                        break # Found the Literal, no need to check further
                        except Exception as e:
                            logging.error(f"Could not introspect AlignedClause.original_clause_type annotation: {e}")
                                
                        if isinstance(received_type, str):
                            normalized_type = received_type.strip().lower()
                            if allowed_literals and normalized_type in allowed_literals:
                                validated_type = normalized_type
                            else:
                                # Log warning only if we could actually get the allowed types
                                allowed_str = f"Allowed: {allowed_literals}" if allowed_literals else "Could not determine allowed types."
                                logging.warning(f"Invalid original_clause_type '{received_type}' received for {clause.clause_id}. Setting to None. {allowed_str}")
                        elif received_type is not None:
                            logging.warning(f"Unexpected type for original_clause_type: {type(received_type)} for {clause.clause_id}. Setting to None.")
                                
                        clause.original_clause_type = validated_type
                        logging.debug(f"  Updated clause {clause.clause_id}: aligned='{clause.aligned_original_clause_segment}', type='{clause.original_clause_type}'")
                    else:
                        logging.warning(f"Could not find clause with id '{clause.clause_id}' in AS Unit {unit['as_unit_id']} to apply alignment.")
        # --- 6. Final Filtering Step (AS Unit Length) ---
        min_as_unit_words = 3
        final_filtered_units = [
            unit for unit in all_segmented_units
            if count_words(unit.get('as_unit_text', '')) > min_as_unit_words # Change >= to >
        ]
        items_filtered_out_final = len(all_segmented_units) - len(final_filtered_units)
        if items_filtered_out_final > 0:
            logger.info(f"Filtered out {items_filtered_out_final} AS units with < {min_as_unit_words} words in 'as_unit_text'.")
        logger.info(f"Final AS unit count for batch: {len(final_filtered_units)}")

        # TODO: Add Clause Analysis steps here later, operating on final_filtered_units
        logger.info(f"Final AS unit count for batch: {len(final_filtered_units)}")

        # Convert final dictionaries to PreprocessedASUnit objects
        preprocessed_units_to_return: List[PreprocessedASUnit] = []
        for unit_dict in final_filtered_units:
            try:
                # Ensure clauses are AlignedClause objects before creating the parent object
                clauses_list = unit_dict.get('clauses', [])
                if not all(isinstance(c, AlignedClause) for c in clauses_list):
                    logger.warning(f"Skipping AS unit {unit_dict.get('as_unit_id')} due to unexpected clause type. Expected AlignedClause.")
                    continue

                preprocessed_unit = PreprocessedASUnit(
                    as_unit_id=unit_dict['as_unit_id'], # Use direct access assuming keys exist after filtering
                    original_utterance_id=unit_dict['original_utterance_id'],
                    original_input_text=unit_dict['original_input_text'],
                    as_unit_text=unit_dict['as_unit_text'],
                    aligned_original_text=unit_dict.get('aligned_original_text'), # This can be None
                    clauses=clauses_list,
                    context=None # Context added later in main.py
                )
                preprocessed_units_to_return.append(preprocessed_unit)
            except KeyError as ke:
                 logger.error(f"Missing expected key {ke} when creating PreprocessedASUnit for data: {unit_dict}. Skipping unit.")
            except Exception as e: # Catch Pydantic validation errors or others
                 logger.error(f"Failed to create PreprocessedASUnit for AS unit ID {unit_dict.get('as_unit_id', 'UNKNOWN')}: {e}. Skipping unit.")

        return preprocessed_units_to_return 

    except Exception as e:
        logger.exception(f"[Preprocessing Module] Error during preprocessing batch execution: {e}")
        # Depending on requirements, you might want to return an empty list,
        # partially processed data, or raise the exception.
        # Returning empty list for now to avoid downstream failures on partial data.
        return []

# Example Usage (for testing or direct execution)
if __name__ == "__main__":
    pass
