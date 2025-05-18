#!/usr/bin/env python3
"""
Test script for report card generation integration.
This script simulates the API flow for report card generation without needing to start the full frontend.
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

async def test_report_card_generation():
    """Test the report card generation independently using existing data files."""
    try:
        # Use existing data files that we know are valid
        output_dir = os.path.join(project_root, "clustering_output")
        
        # Use the existing test_api data files we created earlier
        analysis_data_path = os.path.join(output_dir, "test_api_report_card_data_analysis.json")
        recommendation_data_path = os.path.join(output_dir, "test_api_report_card_data_recommendation.json")
        report_card_output_path = os.path.join(output_dir, "test_api_report_card_test.md")
        
        # Verify the data files exist
        if not os.path.exists(analysis_data_path) or not os.path.exists(recommendation_data_path):
            logger.error(f"Required data files not found. Please run the clustering pipeline first.")
            return False
        
        logger.info(f"Using analysis data: {analysis_data_path}")
        logger.info(f"Using recommendation data: {recommendation_data_path}")
        logger.info(f"Output will be written to: {report_card_output_path}")
        
        # Import the report card generation function directly
        from clustering_analysis.generate_report_card import generate_report_card
        
        # Run the report card generation
        logger.info("Starting report card generation...")
        start_time = time.time()
        
        await generate_report_card(
            analysis_data_path=analysis_data_path,
            recommendation_data_path=recommendation_data_path,
            output_path=report_card_output_path
        )
        
        total_time = time.time() - start_time
        logger.info(f"Report card generation completed in {total_time:.2f}s")
        
        # Check if the report card was generated
        if os.path.exists(report_card_output_path):
            logger.info(f"Report card generated successfully: {report_card_output_path}")
            
            # Print the first few lines of the report card
            with open(report_card_output_path, "r") as f:
                content = f.read(500)  # Read first 500 characters
                logger.info(f"Report card preview:\n{content}...")
            
            # Also test the API endpoint functionality
            logger.info("Testing the report card API endpoint functionality...")
            
            # Simulate the API endpoint logic
            conversation_id = "test_api"
            found_report_card = None
            
            for file in os.listdir(output_dir):
                if file.startswith(conversation_id) and file.endswith("_report_card_test.md"):
                    found_report_card = os.path.join(output_dir, file)
                    break
            
            if found_report_card:
                logger.info(f"API endpoint would successfully find the report card: {found_report_card}")
            else:
                logger.error(f"API endpoint would fail to find the report card")
        else:
            logger.error(f"Report card was not generated: {report_card_output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_report_card_generation())
