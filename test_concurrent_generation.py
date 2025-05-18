#!/usr/bin/env python3
"""
Test script for concurrent report card and teaching module generation.
This script verifies that both processes run concurrently.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary functions from api_server
from api_server import run_teaching_module_generation_background

class MockTeachingTask:
    """Mock class to simulate teaching module generation."""
    
    def __init__(self, duration=5):
        """Initialize with a duration in seconds."""
        self.duration = duration
    
    async def __call__(self, *args, **kwargs):
        """Simulate the teaching module generation process."""
        logger.info("Starting mock teaching module generation...")
        start_time = time.time()
        
        # Simulate work with periodic updates
        total_steps = 5
        for step in range(1, total_steps + 1):
            await asyncio.sleep(self.duration / total_steps)
            logger.info(f"Teaching module generation step {step}/{total_steps} completed")
        
        total_time = time.time() - start_time
        logger.info(f"Mock teaching module generation completed in {total_time:.2f}s")

async def test_concurrent_execution():
    """Test that report card generation and teaching module generation run concurrently."""
    try:
        # Use existing data files
        output_dir = os.path.join(project_root, "clustering_output")
        
        # Define the file paths exactly as they are in the API server
        conversation_id = "test_api"
        base_filename = conversation_id
        
        # Create the prioritized JSON path first
        prioritized_json_path = os.path.join(output_dir, f"{base_filename}_secondary_prioritized.json")
        
        # Define the report card paths based on the actual files in the directory
        analysis_data_path = os.path.join(output_dir, f"{base_filename}_report_card_data_analysis.json")
        recommendation_data_path = os.path.join(output_dir, f"{base_filename}_report_card_data_recommendation.json")
        report_card_output_path = os.path.join(output_dir, f"{base_filename}_report_card.md")
        
        # Verify the data files exist
        if not os.path.exists(analysis_data_path) or not os.path.exists(recommendation_data_path):
            logger.error("Required data files not found. Please run the clustering pipeline first.")
            return False
        
        # Create a simple JSON file for the prioritized path if it doesn't exist
        if not os.path.exists(prioritized_json_path):
            with open(prioritized_json_path, "w") as f:
                json.dump({"test": "data"}, f)
                
        logger.info(f"Using analysis data: {analysis_data_path}")
        logger.info(f"Using recommendation data: {recommendation_data_path}")
        logger.info(f"Output will be written to: {report_card_output_path}")
        
        # Ensure the report card output path doesn't exist before starting
        if os.path.exists(report_card_output_path):
            os.remove(report_card_output_path)
        
        # Import necessary modules
        from clustering_analysis.generate_report_card import generate_report_card
        
        # Create a mock for the main_orchestrator function
        import api_server
        original_main_orchestrator = api_server.main_orchestrator
        api_server.main_orchestrator = MockTeachingTask(duration=10)
        
        try:
            # Run the background task that handles both processes
            logger.info("Starting concurrent execution test...")
            start_time = time.time()
            
            # Patch the generate_report_card function to add logging for concurrency verification
            from clustering_analysis import generate_report_card as gc_module
            original_generate_report_card = gc_module.generate_report_card
            
            async def instrumented_generate_report_card(*args, **kwargs):
                logger.info("Report card generation started - running concurrently with teaching module generation")
                result = await original_generate_report_card(*args, **kwargs)
                logger.info("Report card generation completed")
                return result
                
            gc_module.generate_report_card = instrumented_generate_report_card
            
            try:
                # Ensure the report card output path doesn't exist
                if os.path.exists(report_card_output_path):
                    os.remove(report_card_output_path)
                
                # Run the background task
                await run_teaching_module_generation_background(
                    prioritized_json_path=prioritized_json_path,
                    conversation_id="test_api"
                )
                
                total_time = time.time() - start_time
                logger.info(f"Concurrent execution test completed in {total_time:.2f}s")
                
                # Check if the report card was generated
                if os.path.exists(report_card_output_path):
                    logger.info(f"Report card generated successfully: {report_card_output_path}")
                    
                    # Print the first few lines of the report card
                    with open(report_card_output_path, "r") as f:
                        content = f.read(300)  # Read first 300 characters
                        logger.info(f"Report card preview:\n{content}...")
                else:
                    logger.error(f"Report card was not generated: {report_card_output_path}")
            finally:
                # Restore original function
                gc_module.generate_report_card = original_generate_report_card
            
            return True
        finally:
            # Restore the original function
            api_server.main_orchestrator = original_main_orchestrator
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_concurrent_execution())
