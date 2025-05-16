import argparse
import asyncio
import json
import logging
import logging.config
import os
import math
import time
import warnings
import subprocess
import sys
from typing import List, Generator, Dict, Any, Tuple
from pydantic import BaseModel

# Filter out specific deprecation warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

# Disable LangSmith tracing to avoid rate limit warnings
# Only set if not already configured in .env file
if "LANGCHAIN_TRACING_V2" not in os.environ:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
if "LANGCHAIN_ENDPOINT" not in os.environ:
    os.environ["LANGCHAIN_ENDPOINT"] = ""  # Empty string to avoid any fallback
from analyzer_service.schemas import (
    InputUtterance,
    AlignedASUnit,
    PreprocessedASUnit,
    AlignedClause,
    PreprocessingOutput,
    ContextUtterance
)
from analyzer_service.utils import contains_korean

# --- Configuration --- 
LOGGING_CONFIG_PATH = 'logging.conf'

# --- Setup Logging --- 
def setup_logging():
    """Sets up logging based on the configuration file."""
    # Construct path relative to this script's location
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOGGING_CONFIG_PATH)
    if os.path.exists(log_file_path):
        try:
            logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
            logging.info("Logging configured from file.")
        except Exception as e:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(f"Error configuring logging from {log_file_path}: {e}. Using basic config.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning(f"Logging configuration file not found at {log_file_path}. Using basic config.")

# Create a logger for this module that will use the analyzer_service namespace
logger = logging.getLogger('analyzer_service')

# --- Import Stage Processing Functions --- 
try:
    from analyzer_service.preprocessing import (
        run_preprocessing_batch,
        initialize_clause_identification_chain,
        parse_clause_identification_output,
        initialize_clause_alignment_chain,
        parse_clause_alignment_output
    )
    from analyzer_service.analysis import run_analysis_batch, _run_correction_chain_on_batch
    from analyzer_service.teaching import run_teaching_batch
except ImportError as e:
    logging.error(f"Failed to import necessary processing functions: {e}", exc_info=True)
    logging.critical("Core processing modules missing. Cannot proceed.")
    exit(1)

# --- Data Handling --- 
def load_utterances(input_path: str) -> Tuple[List[InputUtterance], List[InputUtterance]]:
    """Loads all utterances from a JSON file and separately returns student utterances."""
    abs_input_path = os.path.abspath(input_path)
    logging.info(f"Loading utterances from: {abs_input_path}")
    try:
        with open(abs_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        utterance_list = data.get('utterances')
        if utterance_list is None:
            logging.error(f"'utterances' key not found in JSON data: {abs_input_path}")
            raise ValueError("'utterances' key not found in JSON data")
        if not isinstance(utterance_list, list):
             logging.error(f"Value associated with 'utterances' key is not a list in {abs_input_path}")
             raise ValueError("Value associated with 'utterances' key is not a list")

        all_parsed_utterances: List[InputUtterance] = []
        student_only_utterances: List[InputUtterance] = []

        for item in utterance_list:
            if isinstance(item, dict):
                try:
                    # Create InputUtterance for every item for the full list
                    full_utt = InputUtterance(**item)
                    all_parsed_utterances.append(full_utt)
                    
                    # If it's a student, also add to the student-only list
                    if item.get('speaker') == 'student':
                        student_only_utterances.append(full_utt)
                except Exception as e: # More specific: pydantic.ValidationError
                    logging.warning(f"Skipping utterance due to parsing error: {item}. Error: {e}")
        
        logging.info(f"Loaded {len(all_parsed_utterances)} total utterances from file.")
        if not student_only_utterances:
            logging.warning("No utterances with speaker='student' found in the input file.")
        else:
            logging.info(f"Identified {len(student_only_utterances)} student utterances for processing.")
            
        return all_parsed_utterances, student_only_utterances
    except FileNotFoundError:
        logging.error(f"Input file not found: {abs_input_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file: {abs_input_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {abs_input_path}: {e}")
        raise

def create_batches(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yields successive n-sized chunks from data."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        logging.error(f"Batch size must be a positive integer, got: {batch_size}")
        raise ValueError("Batch size must be a positive integer.")
    total_batches = math.ceil(len(data) / batch_size)
    logging.info(f"Creating {total_batches} batches of size {batch_size} from {len(data)} items.")
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def save_results(results: Dict[str, Any], output_path: str):
    """Saves the aggregated results to a JSON file."""
    abs_output_path = os.path.abspath(output_path)
    logging.info(f"Saving {len(results)} results to: {abs_output_path}")
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        
        with open(abs_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4) # Save the dict directly
        logging.info(f"Successfully saved results to {abs_output_path}")
    except TypeError as e:
        logging.error(f"Failed to serialize results to JSON: {e}")

# --- Refactored Clause Identification --- #
def run_clause_identification_on_batch(batch_as_units: List[PreprocessedASUnit], chain, parser) -> List[PreprocessedASUnit]:
    """Runs clause identification on a batch of PreprocessedASUnit objects.

    Args:
        batch_as_units: List of PreprocessedASUnit objects.
        chain: Initialized LangChain for clause identification.
        parser: Function to parse the raw output from the chain.

    Returns:
        The original list of PreprocessedASUnit objects, modified in-place
        to include identified clauses or error status.
    """
    logger = logging.getLogger(__name__)
    if not batch_as_units:
        logger.info("[Clause ID Step] Empty batch received, skipping.")
        return []

    # Prepare batch input for clause identification using attribute access
    # Filter units that actually have text to process
    units_to_process = [unit for unit in batch_as_units if unit.as_unit_text]
    if not units_to_process:
        logger.warning("[Clause ID Step] No AS units with text found in batch to process.")
        # Mark units without text as skipped (if they weren't already)
        for unit in batch_as_units:
            if not unit.as_unit_text:
                 unit.clauses = [] # Ensure clauses list exists
                 # Optionally set a status if needed, though None might imply skip
        return batch_as_units # Return original list

    batch_input_clause_id = [
        {"as_unit_text": unit.as_unit_text}
        for unit in units_to_process
    ]

    logger.info(f"[Clause ID Step] Invoking chain batch with {len(batch_input_clause_id)} AS units...")
    start_time = time.time()
    try:
        # Assuming chain.batch correctly handles the list of dicts input
        clause_id_raw_results = chain.batch(batch_input_clause_id)
        duration = time.time() - start_time
        logger.info(f"[Clause ID Step] Chain batch call completed in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"[Clause ID Step] Error during clause identification batch call: {e}", exc_info=True)
        # Mark units *that were attempted* as failed for this step
        for unit in units_to_process:
            unit.clauses = [] # Clear/ensure empty clauses
            # unit.status = "clause_identification_failed_error" # Optionally set status
        return batch_as_units # Return original list

    # Check result count against the number of units *sent* to the chain
    if len(clause_id_raw_results) != len(units_to_process):
        logger.error(f"[Clause ID Step] Result count mismatch: {len(units_to_process)} inputs sent vs {len(clause_id_raw_results)} results received. Marking processed units as skipped.")
        for unit in units_to_process:
             unit.clauses = []
             # unit.status = "clause_identification_skipped_mismatch"
        return batch_as_units

    # Process results and add to the original unit objects
    for i, unit in enumerate(units_to_process):
        raw_result = clause_id_raw_results[i]
        try:
            # The parser now returns List[str]
            clause_texts = parser(raw_result)
        except Exception as e:
            logger.error(f"[Clause ID Step] Error parsing result for AS unit '{unit.as_unit_id}': {e}", exc_info=True)
            unit.clauses = []
            # unit.status = "clause_parsing_failed"
            continue # Move to the next unit

        clauses_data = []
        # Iterate over the list of clause text strings
        for index, text in enumerate(clause_texts):
            if not text: # Skip if clause text is empty after parsing
                logger.warning(f"[Clause ID Step] Skipping empty clause_text for unit '{unit.as_unit_id}' at index {index}.")
                continue

            # Create AlignedClause objects using the text
            # Initialize fields to be populated by alignment step as None
            clause = AlignedClause(
                clause_id=f"{unit.as_unit_id}-c{index}",
                clause_text=text.strip(), # Text from identification
                aligned_original_clause_segment=None, # To be filled by alignment
                is_korean=None, # To be filled by alignment
                original_clause_type=None # To be filled by alignment
            )
            clauses_data.append(clause)

        # Update the original unit object directly
        unit.clauses = clauses_data
        # logger.debug(f"[Clause ID Step] Added {len(clauses_data)} clauses to AS unit '{unit.as_unit_id}'")
        # Optionally update status if needed

    # Return the original list, now modified with clauses
    return batch_as_units

# --- Clause Alignment --- #
def run_clause_alignment_on_batch(
    batch_as_units: List[PreprocessedASUnit],
    chain, # The alignment chain
    parser # The alignment parser function
) -> List[PreprocessedASUnit]:
    """Runs clause alignment on a batch of PreprocessedASUnit objects.

    Updates the AlignedClause objects in-place with alignment results.

    Args:
        batch_as_units: List of PreprocessedASUnit objects with clauses identified.
        chain: Initialized LangChain for clause alignment.
        parser: Function to parse the raw output from the chain.

    Returns:
        The original list of PreprocessedASUnit objects, modified in-place.
    """
    logger = logging.getLogger(__name__)
    if not batch_as_units:
        logger.info("[Alignment Step] Empty batch received, skipping.")
        return []

    # Prepare batch input - filter units that have clauses to align
    units_to_align = []
    batch_input_alignment = []
    unit_clause_map = {} # Track which unit owns which clauses for easier result mapping

    for unit in batch_as_units:
        if unit.clauses and unit.aligned_original_text:
            units_to_align.append(unit)
            clauses_json_list = []
            for clause in unit.clauses:
                # Only pass necessary info to the alignment prompt
                clauses_json_list.append({
                    "clause_id": clause.clause_id,
                    "clause_text": clause.clause_text
                })
                unit_clause_map[clause.clause_id] = unit # Map clause_id back to its unit
            
            batch_input_alignment.append({
                "aligned_original_text": unit.aligned_original_text,
                "clauses": json.dumps(clauses_json_list, ensure_ascii=False)
            })
        else:
            if not unit.clauses:
                logger.debug(f"[Alignment Step] Skipping AS unit '{unit.as_unit_id}' because it has no clauses.")
            if not unit.aligned_original_text:
                 logger.debug(f"[Alignment Step] Skipping AS unit '{unit.as_unit_id}' because it has no aligned_original_text.")
            # Ensure clauses list exists even if empty
            if not hasattr(unit, 'clauses') or unit.clauses is None:
                unit.clauses = []

    if not batch_input_alignment:
        logger.warning("[Alignment Step] No AS units with both clauses and original text found in batch.")
        return batch_as_units

    logger.info(f"[Alignment Step] Invoking chain batch with {len(batch_input_alignment)} AS units...")
    start_time = time.time()
    try:
        # Assuming chain.batch correctly handles the list of dicts input
        alignment_raw_results = chain.batch(batch_input_alignment)
        duration = time.time() - start_time
        logger.info(f"[Alignment Step] Chain batch call completed in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"[Alignment Step] Error during alignment batch call: {e}", exc_info=True)
        # Optionally mark units as failed for this step
        return batch_as_units # Return original list, clauses won't be updated

    # Check result count against the number of units *sent* to the chain
    if len(alignment_raw_results) != len(units_to_align):
        logger.error(f"[Alignment Step] Result count mismatch: {len(units_to_align)} inputs sent vs {len(alignment_raw_results)} results received.")
        return batch_as_units

    # Process results and update the original unit objects
    # Create a single map of all clause_id -> alignment_data across the batch
    all_clauses_alignment_map: Dict[str, Dict[str, Any]] = {}
    failed_parses = 0
    for i, raw_result in enumerate(alignment_raw_results):
        unit_id_for_logging = units_to_align[i].as_unit_id # Get ID for logging
        try:
            parsed_alignments = parser(raw_result) # Expect List[Dict]
            for alignment_data in parsed_alignments:
                clause_id = alignment_data.get('clause_id')
                if clause_id:
                    all_clauses_alignment_map[clause_id] = alignment_data
                else:
                    logger.warning(f"[Alignment Step] Parsed alignment data missing clause_id for unit '{unit_id_for_logging}'. Item: {alignment_data}")
        except Exception as e:
            logger.error(f"[Alignment Step] Error parsing alignment result for AS unit '{unit_id_for_logging}': {e}", exc_info=True)
            failed_parses += 1
            # Continue processing other units, but this one's clauses won't be updated

    if failed_parses > 0:
        logger.warning(f"[Alignment Step] Failed to parse alignment results for {failed_parses} units.")

    # Update AlignedClause objects using the map
    updated_clause_count = 0
    missing_alignment_count = 0
    for unit in units_to_align: # Iterate only through units that were sent for alignment
        for clause in unit.clauses:
            alignment_data = all_clauses_alignment_map.get(clause.clause_id)
            if alignment_data:
                aligned_segment = alignment_data.get('aligned_original_clause_segment')
                clause.aligned_original_clause_segment = aligned_segment
                # --- Debugging is_korean --- #
                is_korean_result = contains_korean(aligned_segment)
                logger.debug(f"[Alignment Debug] Clause ID: {clause.clause_id}, Segment: '{aligned_segment}', contains_korean result: {is_korean_result}")
                # --- End Debugging --- #
                # Calculate is_korean using the helper function
                clause.is_korean = is_korean_result
                clause.original_clause_type = alignment_data.get('original_clause_type')
                updated_clause_count += 1
            else:
                # Alignment data wasn't returned or parsed for this specific clause_id
                logger.warning(f"[Alignment Step] Alignment data not found for clause_id '{clause.clause_id}' in unit '{unit.as_unit_id}'. Leaving fields as None.")
                missing_alignment_count += 1
                # Fields remain None as initialized

    logger.info(f"[Alignment Step] Updated {updated_clause_count} clauses with alignment data. {missing_alignment_count} clauses had missing alignment info.")

    # Return the original list, now modified with alignment details
    return batch_as_units

# --- Main Orchestration --- #
async def main():
    """Main function to orchestrate the loading, processing, and saving."""
    setup_logging() # Initialize logging first
    
    parser = argparse.ArgumentParser(description="Analyzer Service v5.0 - Batch Processing Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file containing utterances.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file for results.")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Number of utterances to process per batch.")
    
    args = parser.parse_args()
    
    logging.info("Starting Analyzer Service pipeline...")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Batch size: {args.batch_size}")

    try:
        # 1. Load all data
        full_conversation_utterances, student_utterances_for_processing = load_utterances(args.input)
        
        if not student_utterances_for_processing:
             logging.warning("Input file loaded successfully, but contained no student utterances for processing. Exiting.")
             return

        # Create utterance index map from the full conversation
        utterance_index_map: Dict[str, int] = {utt.id: i for i, utt in enumerate(full_conversation_utterances)}

        # 2. Create batches from student-only utterances
        utterance_batches = create_batches(student_utterances_for_processing, args.batch_size)
        
        # --- Initialize Components Once --- #
        try:
            logger = logging.getLogger(__name__)
            logger.info("Initializing Clause Identification components...")
            clause_id_chain = initialize_clause_identification_chain()
            clause_id_parser = parse_clause_identification_output # Assign function itself
            logger.info("Clause Identification components initialized.")

            # --- Initialize Clause Alignment Chain & Parser ---
            clause_alignment_chain = initialize_clause_alignment_chain()
            clause_align_parser = parse_clause_alignment_output # Assign the new parser function
            logger.info("Clause Alignment chain and parser initialized.")

            # Initialize other chains/components here if needed (e.g., for correction, analysis)
        except Exception as e:
            logging.critical(f"Failed to initialize pipeline components: {e}", exc_info=True)
            exit(1)

        # 3. Process batches sequentially
        all_results: List[Dict[str, Any]] = [] # Updated to store final results per AS unit
        batch_num = 0
        total_start_time = time.time() # Overall pipeline start time

        for batch in utterance_batches:
            batch_num += 1
            batch_start_time = time.time() # <<< START timer for this batch
            logger.info(f"--- Processing Batch {batch_num} --- ")

            try:
                # 1. Preprocessing Stage
                preprocessed_batch_results = run_preprocessing_batch(batch)
                # Directly append the list of AlignedASUnit objects
                processed_units_with_context: List[PreprocessedASUnit] = []
                for unit in preprocessed_batch_results:
                    original_utterance_id = unit.original_utterance_id
                    if original_utterance_id in utterance_index_map:
                        current_index = utterance_index_map[original_utterance_id]
                        start_index = max(0, current_index - 3) # Get index for context start
                        # Use full_conversation_utterances to get context including interviewer
                        context_utterances_list = full_conversation_utterances[start_index:current_index]
                        
                        # Create ContextUtterance objects
                        unit.context = [ContextUtterance(speaker=u.speaker, text=u.text) for u in context_utterances_list]
                    else:
                        logging.warning(f"Original utterance ID '{original_utterance_id}' for AS unit '{unit.as_unit_id}' not found in index map. Skipping context.")
                        unit.context = [] # Or None, depending on preference
                    processed_units_with_context.append(unit)

                # 2. Clause Identification Stage
                logger.info("Running clause identification...")
                # Pass the list of PreprocessedASUnit objects with context
                # Use the pre-initialized chain and parser
                preprocessed_units_with_clauses = run_clause_identification_on_batch(
                    processed_units_with_context, clause_id_chain, clause_id_parser
                )
                logger.info("Clause identification completed.")

                # 3. Clause Alignment
                preprocessed_units_with_clauses = run_clause_alignment_on_batch(
                    preprocessed_units_with_clauses, 
                    clause_alignment_chain, 
                    clause_align_parser # Pass the parser
                )
                logger.info("Clause alignment completed.")

                # --- Combine Preprocessing, Analysis, and Teaching --- #
                # 1. Preprocessing steps (already done above, resulting in preprocessed_units_with_clauses)
                #    This list now contains PreprocessedASUnit objects ready for analysis.
                
                # 2. Analysis Stage
                # Pass the fully preprocessed batch to the main analysis function
                logger.info("Running analysis stage (correction, accuracy, pattern analysis)...")
                # Await the async analysis function
                analysis_results_batch = await run_analysis_batch(preprocessed_units_with_clauses)
                logger.info("Analysis stage completed.")

                # 3. Teaching Stage (Placeholder - Requires analysis_results_batch)
                # teaching_recommendations_batch = run_teaching_batch(analysis_results_batch)
                # logger.info("Teaching stage completed.")
                
                # 4. Aggregation
                # For now, aggregate the analysis results. Teaching results would be merged here.
                all_results.extend(analysis_results_batch) 

            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {e}")

            batch_end_time = time.time() # <<< END timer for this batch
            batch_duration = batch_end_time - batch_start_time
            logger.info(f"--- Batch {batch_num} processed in {batch_duration:.2f} seconds ---") # <<< LOG duration

        logger.info(f"--- Finished processing all batches --- ")

        total_end_time = time.time() # Overall pipeline end time
        total_duration = total_end_time - total_start_time
        logger.info(f"--- All {batch_num} batches processed in {total_duration:.2f} seconds ---")

        # --- Save Results --- 
        if all_results: 
            final_results = {f"result_{i}": result for i, result in enumerate(all_results)}
            final_output_path = os.path.abspath(args.output) # Store the absolute path
            save_results(final_results, final_output_path)
            logging.info(f"--- Analysis Pipeline Complete --- Output saved to {final_output_path} ---")

            # --- Automatically run Clustering Analysis --- #
            if final_output_path:
                logging.info("--- Starting Clustering Analysis Pipeline ---")
                clustering_script_module = "clustering_analysis.run_clustering_pipeline"
                # Use sys.executable to ensure the same Python interpreter is used
                command = [sys.executable, "-m", clustering_script_module, final_output_path]
                project_root_dir = os.path.dirname(__file__) # Get directory of main.py
                logging.info(f"Running command: {' '.join(command)} in directory: {project_root_dir}")
                try:
                    # Run clustering analysis synchronously after main analysis completes
                    result = subprocess.run(
                        command, 
                        check=True,            # Raise exception on non-zero exit code
                        capture_output=True,   # Capture stdout/stderr
                        text=True,             # Decode stdout/stderr as text
                        cwd=project_root_dir   # Set working directory to project root
                    )
                    logging.info(f"Clustering analysis completed successfully.")
                    # Log stdout/stderr at DEBUG level to avoid cluttering INFO logs
                    logging.debug(f"Clustering stdout:\n{result.stdout}") 
                    if result.stderr:
                        logging.warning(f"Clustering stderr:\n{result.stderr}") # Log stderr as warning
                except subprocess.CalledProcessError as e:
                    # Log detailed error information if the subprocess fails
                    logging.error(f"Error running clustering analysis script (module: {clustering_script_module}):")
                    logging.error(f"  Command: {' '.join(e.cmd)}")
                    logging.error(f"  Return code: {e.returncode}")
                    logging.error(f"  Stdout:\n{e.stdout}")
                    logging.error(f"  Stderr:\n{e.stderr}")
                except FileNotFoundError:
                    # Log error if the Python executable or module isn't found
                    logging.error(f"Error: Clustering analysis module '{clustering_script_module}' or Python executable '{sys.executable}' not found.")
                except Exception as e:
                     # Catch any other unexpected errors
                     logging.error(f"An unexpected error occurred during clustering analysis execution: {e}", exc_info=True)
        else:
             # This case should ideally not happen if aggregated_results is true
             logging.warning("Clustering analysis skipped as final output path was not set.")

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}") 

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
