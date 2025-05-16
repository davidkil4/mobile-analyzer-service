"""
Main orchestrator script for the teaching module generation pipeline.

This script connects the following stages:
1. filter_and_split.py: Filters and splits the initial large JSON analysis file.
2. batch_focus_selection.py: Determines the focus type (grammar, pattern, vocab) for utterances.
3. generate_teaching_modules.py: Generates and validates teaching modules (explanations, problems) based on focus type.
   Note: Validation is now integrated directly into the generate_teaching_modules.py script using batch validation.
"""
import os
import asyncio
import time
import logging
import concurrent.futures
import sys
import argparse
import shutil  # Clear old outputs to avoid stale files

# --- Setup sys.path for correct imports ---
# Get the absolute path of the directory containing the current script (teaching_module_generation)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (project root, e.g., analyzer_service 6.0)
PROJECT_ROOT_FOR_IMPORTS = os.path.dirname(SCRIPT_DIR)
# Add the project root to sys.path if it's not already there
if PROJECT_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_IMPORTS)

# --- Import functions from the individual pipeline scripts ---
from teaching_module_generation.filter_and_split import filter_and_split
from teaching_module_generation.batch_focus_selection import (
    load_api_key as bfs_load_api_key, 
    process_focus_selection_for_file as bfs_process_file,
    filter_by_types as bfs_filter_by_types, 
    write_output as bfs_write_output
)
from teaching_module_generation.korean_focus_selection import process_korean_focus_selection
from teaching_module_generation.generate_teaching_modules import main as gtm_generate_all_teaching_modules
# Validation is now integrated into generate_teaching_modules.py, so we don't need to import run_validation_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration: File Paths ---
# Adjust these paths if your directory structure is different.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # teaching_module_generation directory
PROJECT_ROOT = os.path.dirname(BASE_DIR) # analyzer_service 6.0 directory

# Default input file (will be overridden by command-line argument if provided)
DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, "clustering_output", "660_golf_analysis_secondary_prioritized.json")

# Stage 1 Inputs/Outputs
# INITIAL_INPUT_FILE will be set in main_orchestrator() based on command-line args
STAGE1_OUTPUT_DIR = os.path.join(BASE_DIR, "output_files")
STAGE1_OUTPUT_PART1 = os.path.join(STAGE1_OUTPUT_DIR, "teaching_input_part1.json")
STAGE1_OUTPUT_PART2 = os.path.join(STAGE1_OUTPUT_DIR, "teaching_input_part2.json")
STAGE1_OUTPUT_KOREAN_INPUT = os.path.join(STAGE1_OUTPUT_DIR, "korean_teaching_input.json")

# Stage 2 Inputs/Outputs (Inputs are outputs from Stage 1)
STAGE2_OUTPUT_DIR = os.path.join(STAGE1_OUTPUT_DIR, "output_files") # Subdirectory: output_files/output_files/
STAGE2_OUTPUT_GRAMMAR = os.path.join(STAGE2_OUTPUT_DIR, "grammar.json")
STAGE2_OUTPUT_PATTERNS = os.path.join(STAGE2_OUTPUT_DIR, "patterns.json")
STAGE2_OUTPUT_VOCAB_SMALL_GRAMMAR = os.path.join(STAGE2_OUTPUT_DIR, "vocabulary_and_small_grammar.json")
STAGE2_OUTPUT_KOREAN_WORD = os.path.join(STAGE2_OUTPUT_DIR, "korean_word_focused.json")
STAGE2_OUTPUT_KOREAN_PHRASE = os.path.join(STAGE2_OUTPUT_DIR, "korean_phrase_focused.json")

# Stage 3 Inputs/Outputs (Inputs are outputs from Stage 2)
# The output directory now contains subdirectories for validated and rejected modules
STAGE3_OUTPUT_DIR = os.path.join(STAGE1_OUTPUT_DIR, "teaching_module_outputs_new") # Subdirectory: output_files/teaching_module_outputs_new/
# Validated modules will be in STAGE3_OUTPUT_DIR/validated/
# Rejected modules will be in STAGE3_OUTPUT_DIR/rejected/
# Validation reports will be in STAGE3_OUTPUT_DIR/validation_reports/

async def main_orchestrator(input_file=None):
    """Runs the entire teaching module generation pipeline.
    
    Args:
        input_file (str, optional): Path to the input JSON file. If not provided, uses the default.
    """
    start_time_total = time.time()
    logger.info("--- STARTING TEACHING MODULE PIPELINE ---")
    
    # Set the input file path based on the argument or use default
    INITIAL_INPUT_FILE = input_file if input_file else DEFAULT_INPUT_FILE
    logger.info(f"Using input file: {INITIAL_INPUT_FILE}")

    # Clear old intermediate outputs from Stage 2
    if os.path.exists(STAGE2_OUTPUT_DIR):
        for fname in os.listdir(STAGE2_OUTPUT_DIR):
            fpath = os.path.join(STAGE2_OUTPUT_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
            else:
                shutil.rmtree(fpath)
    # Prepare Stage 1 output directory
    os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
    # Prepare Stage 2 output directory
    os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
    # Prepare Stage 3 root and clean/create subdirectories
    os.makedirs(STAGE3_OUTPUT_DIR, exist_ok=True)
    for sub in ["validated", "rejected", "validation_reports"]:
        subdir = os.path.join(STAGE3_OUTPUT_DIR, sub)
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
        os.makedirs(subdir, exist_ok=True)

    # --- Create output directories if they don't exist ---
    # os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(STAGE3_OUTPUT_DIR, exist_ok=True)
    # Create subdirectories for validated and rejected modules
    # os.makedirs(os.path.join(STAGE3_OUTPUT_DIR, "validated"), exist_ok=True)
    # os.makedirs(os.path.join(STAGE3_OUTPUT_DIR, "rejected"), exist_ok=True)
    # os.makedirs(os.path.join(STAGE3_OUTPUT_DIR, "validation_reports"), exist_ok=True)

    # --- Stage 1: Filter and Split ---
    logger.info("--- Stage 1: Running Filter and Split ---")
    start_time_s1 = time.time()
    try:
        # Check if input file exists
        if not os.path.exists(INITIAL_INPUT_FILE):
            logger.error(f"Input file not found: {INITIAL_INPUT_FILE}")
            return
            
        filter_and_split(INITIAL_INPUT_FILE, STAGE1_OUTPUT_PART1, STAGE1_OUTPUT_PART2)
        logger.info(f"Filter and Split completed. Outputs: {STAGE1_OUTPUT_PART1}, {STAGE1_OUTPUT_PART2}")
    except Exception as e:
        logger.error(f"Error in Stage 1 (Filter and Split): {e}", exc_info=True)
        return # Stop pipeline if stage 1 fails
    logger.info(f"--- Stage 1 finished in {time.time() - start_time_s1:.2f} seconds ---")

    # --- Stage 1.5: Korean Focus Selection ---
    logger.info("--- Stage 1.5: Running Korean Focus Selection ---")
    start_time_s1_5 = time.time()
    api_key_for_korean_fs = bfs_load_api_key() # Reuse API key loading
    if not api_key_for_korean_fs:
        logger.error("GOOGLE_API_KEY not found. Cannot run Korean Focus Selection.")
        return # Stop pipeline if API key is missing

    try:
        if not os.path.exists(STAGE1_OUTPUT_KOREAN_INPUT):
            logger.warning(f"Korean input file not found: {STAGE1_OUTPUT_KOREAN_INPUT}. Skipping Korean Focus Selection. This may be normal if no Korean utterances were found in Stage 1.")
        else:
            logger.info(f"Processing Korean input file: {STAGE1_OUTPUT_KOREAN_INPUT}")
            # Ensure STAGE2_OUTPUT_DIR exists, as korean_focus_selection expects it for its outputs
            os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)
            process_korean_focus_selection(
                input_path=STAGE1_OUTPUT_KOREAN_INPUT,
                output_dir=STAGE2_OUTPUT_DIR, 
                api_key=api_key_for_korean_fs
            )
            logger.info(f"Korean Focus Selection completed. Outputs should be in {STAGE2_OUTPUT_DIR} (e.g., {STAGE2_OUTPUT_KOREAN_WORD}, {STAGE2_OUTPUT_KOREAN_PHRASE})")
    except Exception as e:
        logger.error(f"Error in Stage 1.5 (Korean Focus Selection): {e}", exc_info=True)
        return # Stop pipeline if this stage fails
    logger.info(f"--- Stage 1.5 finished in {time.time() - start_time_s1_5:.2f} seconds ---")

    # --- Stage 2: Batch Focus Selection (for English/Other utterances) ---
    logger.info("--- Stage 2: Running Batch Focus Selection for English/Other utterances ---")
    stage2_start_time = time.time()
    try:
        bfs_api_key = bfs_load_api_key()
        if not bfs_api_key:
            raise RuntimeError("GOOGLE_API_KEY not found for Batch Focus Selection.")

        # Files from Stage 1 are inputs to Stage 2
        s2_input_files = [STAGE1_OUTPUT_PART1, STAGE1_OUTPUT_PART2]
        processed_data_s2 = [] # To store results from parallel processing

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(s2_input_files))) as executor:
            future_to_file = {executor.submit(bfs_process_file, input_file, bfs_api_key): input_file for input_file in s2_input_files}
            for future in concurrent.futures.as_completed(future_to_file):
                input_filename = future_to_file[future]
                try:
                    data = future.result()
                    if data: # Ensure data is not None (e.g., if bfs_process_file had an internal error)
                        processed_data_s2.append(data)
                        logger.info(f"Successfully processed for focus selection: {input_filename}")
                    else:
                        logger.error(f"Focus selection processing returned no data for {input_filename}. Skipping aggregation for this part.")
                except Exception as exc:
                    logger.error(f"{input_filename} generated an exception during focus selection: {exc}")
                    # Optionally re-raise or handle more gracefully
        
        if not processed_data_s2:
            logger.error("Stage 2: No data was successfully processed by batch_focus_selection. Stopping pipeline.")
            return # Stop if all parts failed

        # Aggregate results from all processed parts
        logger.info("Aggregating focus selection results...")
        aggregated_s2_data = {"metadata": {}, "analysis_zones": []}
        if processed_data_s2:
            # Use metadata from the first successfully processed part (assuming it's consistent)
            aggregated_s2_data["metadata"] = processed_data_s2[0].get("metadata", {})
            for data_part in processed_data_s2:
                aggregated_s2_data["analysis_zones"].extend(data_part.get("analysis_zones", []))
        logger.info(f"Aggregation complete. Total analysis zones: {len(aggregated_s2_data['analysis_zones'])}")

        # Categorize and write aggregated data
        logger.info("Categorizing and writing aggregated focus selection outputs...")
        s2_vocab_data = bfs_filter_by_types(aggregated_s2_data, {"VOCABULARY", "SMALL_GRAMMAR"})
        s2_grammar_data = bfs_filter_by_types(aggregated_s2_data, {"GRAMMAR"})
        s2_patterns_data = bfs_filter_by_types(aggregated_s2_data, {"PATTERN"})

        if s2_vocab_data.get("analysis_zones"):
            bfs_write_output(STAGE2_OUTPUT_VOCAB_SMALL_GRAMMAR, s2_vocab_data)
            logger.info(f"Wrote {STAGE2_OUTPUT_VOCAB_SMALL_GRAMMAR} with {sum(len(z.get('recommendations', [])) for z in s2_vocab_data.get('analysis_zones', []))} items.")
        else:
            logger.info(f"No data for {STAGE2_OUTPUT_VOCAB_SMALL_GRAMMAR}. File not written.")

        if s2_grammar_data.get("analysis_zones"):
            bfs_write_output(STAGE2_OUTPUT_GRAMMAR, s2_grammar_data)
            logger.info(f"Wrote {STAGE2_OUTPUT_GRAMMAR} with {sum(len(z.get('recommendations', [])) for z in s2_grammar_data.get('analysis_zones', []))} items.")
        else:
            logger.info(f"No data for {STAGE2_OUTPUT_GRAMMAR}. File not written.")

        if s2_patterns_data.get("analysis_zones"):
            bfs_write_output(STAGE2_OUTPUT_PATTERNS, s2_patterns_data)
            logger.info(f"Wrote {STAGE2_OUTPUT_PATTERNS} with {sum(len(z.get('recommendations', [])) for z in s2_patterns_data.get('analysis_zones', []))} items.")
        else:
            logger.info(f"No data for {STAGE2_OUTPUT_PATTERNS}. File not written.")

        logger.info(f"Stage 2 completed in {time.time() - stage2_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 2 (Batch Focus Selection): {e}", exc_info=True)
        return # Stop pipeline if stage 2 fails

    # --- Stage 3: Generate and Validate Teaching Modules ---
    logger.info("--- Stage 3: Running Teaching Module Generation with Integrated Validation ---")
    start_time_s3 = time.time()
    try:
        # gtm_generate_all_teaching_modules is an async function
        # It now includes integrated validation with batch processing
        await gtm_generate_all_teaching_modules(
            grammar_input_path=STAGE2_OUTPUT_GRAMMAR,
            patterns_input_path=STAGE2_OUTPUT_PATTERNS,
            vocab_small_grammar_input_path=STAGE2_OUTPUT_VOCAB_SMALL_GRAMMAR,
            output_directory_path=STAGE3_OUTPUT_DIR,
            korean_word_input_path=STAGE2_OUTPUT_KOREAN_WORD, 
            korean_phrase_input_path=STAGE2_OUTPUT_KOREAN_PHRASE
        )
        validated_dir = os.path.join(STAGE3_OUTPUT_DIR, "validated")
        rejected_dir = os.path.join(STAGE3_OUTPUT_DIR, "rejected")
        validation_reports_dir = os.path.join(STAGE3_OUTPUT_DIR, "validation_reports")
        
        # Count the validated and rejected modules
        validated_count = len([f for f in os.listdir(validated_dir) if f.endswith('.json')]) if os.path.exists(validated_dir) else 0
        rejected_count = len([f for f in os.listdir(rejected_dir) if f.endswith('.json')]) if os.path.exists(rejected_dir) else 0
        
        logger.info(f"Teaching Module Generation and Validation completed.")
        logger.info(f"Validated modules: {validated_count} (in {validated_dir})")
        logger.info(f"Rejected modules: {rejected_count} (in {rejected_dir})")
        logger.info(f"Validation reports in: {validation_reports_dir}")
    except Exception as e:
        logger.error(f"Error in Stage 3 (Generate and Validate Teaching Modules): {e}", exc_info=True)
        return # Stop pipeline if stage 3 fails
    logger.info(f"--- Stage 3 finished in {time.time() - start_time_s3:.2f} seconds ---")

    logger.info(f"--- TEACHING MODULE PIPELINE COMPLETED SUCCESSFULLY in {time.time() - start_time_total:.2f} seconds ---")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Teaching Module Generation Pipeline")
    parser.add_argument("-i", "--input", help="Path to the input JSON file (output from clustering analysis)")
    args = parser.parse_args()
    
    # Ensure .env is loaded if scripts rely on it being loaded by their own `load_dotenv()`
    # For generate_teaching_modules and batch_focus_selection, they load their own.
    # filter_and_split does not need .env.
    asyncio.run(main_orchestrator(args.input))
