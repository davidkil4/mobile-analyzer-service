import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Add the project root to the Python path to allow importing analyzer_service
project_root = Path(__file__).parent.parent.resolve() # Go one level up
sys.path.insert(0, str(project_root)) # Add the actual project root

# Assuming ZPD_analyzer is in the same directory
from .ZPD_analyzer import CAFClusterAnalyzer # Import directly as we are in the same dir (relative to execution)

# --- Configuration ---
OUTPUT_DIR = "clustering_output" # Directory to store clustering results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable more verbose logging for debugging
logger.setLevel(logging.DEBUG)

# --- Helper Function ---
def run_script(script_path, args):
    """Runs a Python script as a subprocess."""
    command = [sys.executable, script_path] + args
    logger.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=project_root)
        logger.info(f"Script {script_path} completed successfully.")
        logger.debug(f"Stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script {script_path}:")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(f"Error: Script not found at {script_path}")
        raise

# --- Main Execution ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the clustering analysis pipeline on a given analysis results JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file (e.g., output/analysis_results.json)")
    args = parser.parse_args()

    # Ensure input file exists using the argument
    input_path = Path(args.input_file).resolve() # NEW way using argument, resolve to absolute path

    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    logger.info(f"Using input file: {input_path}")

    # Derive base name from input file
    output_base_name = input_path.stem # e.g., "analysis_results"

    # Create output directory
    output_dir_path = project_root / OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir_path}")

    # Define output file paths using the derived base name
    output_base = output_dir_path / output_base_name # Use derived name
    primary_output_path = f"{output_base}_primary.json"
    state_output_path = f"{output_base}_state.json"
    secondary_txt_path = f"{output_base}_secondary.txt"
    secondary_json_path = f"{output_base}_secondary.json"
    prioritized_json_path = f"{output_base}_secondary_prioritized.json"

    # --- Step 1: ZPD Analyzer (Primary Clustering) ---
    logger.info("--- Running Step 1: ZPD Analyzer (Primary Clustering) ---")
    try:
        # Load and check input data for duplicates before processing
        import json
        with open(input_path, 'r') as f:
            input_data = json.load(f)
        
        # DEBUG LOGGING: Log input data statistics
        logger.info(f"[DEBUG] Loaded {len(input_data)} utterances from {input_path}")
        
        # Check for potential duplicates in input data
        text_counts = {}
        for i, utterance in enumerate(input_data):
            if isinstance(utterance, dict) and 'text' in utterance:
                text = utterance['text']
            elif isinstance(utterance, dict) and 'original' in utterance:
                text = utterance['original']
            else:
                text = f"unknown-format-{i}"
                
            text_snippet = text[:30] if isinstance(text, str) else str(text)[:30]
            if text_snippet in text_counts:
                text_counts[text_snippet].append(i)
            else:
                text_counts[text_snippet] = [i]
        
        # Log any potential duplicates
        for text, indices in text_counts.items():
            if len(indices) > 1:
                logger.warning(f"[DEBUG] Potential duplicate detected: '{text}...' appears at indices {indices}")
        
        analyzer = CAFClusterAnalyzer(str(input_path)) # Use the validated input_path
        logger.info("Extracting features...")
        analyzer.extract_features()
        
        # DEBUG LOGGING: Log feature extraction results
        if hasattr(analyzer, 'features') and analyzer.features is not None and (not hasattr(analyzer.features, '__len__') or len(analyzer.features) > 0):
            logger.info(f"[DEBUG] Extracted {len(analyzer.features)} features")
            
        if analyzer.features is None or (hasattr(analyzer.features, '__len__') and len(analyzer.features) == 0):
            logger.error("Feature extraction failed or resulted in no features. Aborting.")
            sys.exit(1)
        logger.info("Performing clustering...")
        analyzer.perform_clustering()
        logger.info("Determining tendency zone...")
        analyzer.determine_tendency_zone()
        logger.info(f"Generating primary analysis report: {primary_output_path}")
        analyzer.generate_cluster_analysis(primary_output_path)
        logger.info(f"Saving clustering state: {state_output_path}")
        analyzer.save_state(state_output_path)
        logger.info("Step 1 completed successfully.")
    except Exception as e:
        logger.error(f"Error during Step 1 (ZPD Analyzer): {e}", exc_info=True)
        sys.exit(1)

    # --- Step 2: ZPD Regions (Secondary Analysis) ---
    logger.info("--- Running Step 2: ZPD Regions (Secondary Analysis) ---")
    regions_script = project_root / "clustering_analysis" / "ZPD_regions.py" # Correct path
    try:
        run_script(str(regions_script), [primary_output_path, str(output_base)]) # Pass derived output_base
        logger.info("Step 2 completed successfully.")
        logger.info(f"Generated: {secondary_txt_path}")
        logger.info(f"Generated: {secondary_json_path}")
    except Exception as e:
        logger.error(f"Error during Step 2 (ZPD Regions): {e}", exc_info=True)
        sys.exit(1)

    # --- Step 3: ZPD Priority (Prioritization) ---
    logger.info("--- Running Step 3: ZPD Priority (Prioritization) ---")
    priority_script = project_root / "clustering_analysis" / "ZPD_priority.py" # Correct path
    try:
        # ZPD_priority.py expects only the input JSON path according to its usage message
        run_script(str(priority_script), [secondary_json_path])
        logger.info("Step 3 completed successfully.")
        logger.info(f"Generated: {prioritized_json_path}")
    except Exception as e:
        logger.error(f"Error during Step 3 (ZPD Priority): {e}", exc_info=True)
        sys.exit(1)


    logger.info("--- Clustering Analysis Pipeline Test Completed Successfully! ---")

if __name__ == "__main__":
    main()
