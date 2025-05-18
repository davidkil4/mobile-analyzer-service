"""
Test script for report card data generation.

This script simulates what happens in the API server after clustering completes.
It runs the generate_report_card_data.py script on existing clustering output files.
"""
import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.resolve()
    
    # Define input paths (using existing files from clustering_output)
    output_dir_path = project_root / "clustering_output"
    primary_output_path = str(output_dir_path / "660_golf_analysis_new_primary.json")
    secondary_json_path = str(output_dir_path / "660_golf_analysis_new_secondary.json")
    prioritized_json_path = str(output_dir_path / "660_golf_analysis_new_secondary_prioritized.json")
    
    # Define output paths for the report card data files
    report_card_output_path = str(output_dir_path / "test_api_report_card_data.json")
    analysis_output_path = str(output_dir_path / "test_api_analysis_data.json")
    recommendation_output_path = str(output_dir_path / "test_api_recommendation_data.json")
    
    # Log the paths
    logger.info(f"Primary JSON path: {primary_output_path}")
    logger.info(f"Secondary JSON path: {secondary_json_path}")
    logger.info(f"Secondary prioritized JSON path: {prioritized_json_path}")
    logger.info(f"Report card output path: {report_card_output_path}")
    logger.info(f"Analysis output path: {analysis_output_path}")
    logger.info(f"Recommendation output path: {recommendation_output_path}")
    
    # Check if input files exist
    for path in [primary_output_path, secondary_json_path, prioritized_json_path]:
        if not os.path.exists(path):
            logger.error(f"Input file not found: {path}")
            return
    
    # Generate report card data
    try:
        logger.info("Generating report card data...")
        report_card_start_time = time.time()
        
        # Run the report card data generation script
        report_card_script = os.path.join(project_root, "clustering_analysis", "generate_report_card_data.py")
        subprocess.run(
            [sys.executable, report_card_script, 
             primary_output_path, 
             secondary_json_path, 
             prioritized_json_path, 
             report_card_output_path],
            check=True,
            cwd=project_root
        )
        
        report_card_time = time.time() - report_card_start_time
        logger.info(f"Report card data generation completed in {report_card_time:.2f}s")
        
        # Check if output files were created
        for path in [report_card_output_path, analysis_output_path, recommendation_output_path]:
            if os.path.exists(path):
                logger.info(f"Output file created: {path}")
                # Get file size
                file_size = os.path.getsize(path)
                logger.info(f"File size: {file_size} bytes")
                
                # Check if file is valid JSON
                try:
                    import json
                    with open(path, 'r') as f:
                        json.load(f)
                    logger.info(f"File contains valid JSON")
                except json.JSONDecodeError:
                    logger.error(f"File does not contain valid JSON")
            else:
                logger.error(f"Output file not created: {path}")
        
    except Exception as e:
        logger.error(f"Error generating report card data: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
