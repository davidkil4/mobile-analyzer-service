"""
Generate a report card using LLMs.

This script uses Gemini 2.5 Flash to generate a personalized report card for English language learners.
It follows a two-LLM approach:
1. First LLM analyzes complexity and accuracy (Parts 1 & 2)
2. Second LLM provides recommendations (Part 3)

The script takes the analysis and recommendation data files as input and combines the results
into a final report card.
"""
import os
import sys
import json
import time
import logging
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

async def load_prompt(prompt_path):
    """Load a prompt from a file."""
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_path}: {e}")
        return None

async def load_data(data_path):
    """Load data from a JSON file."""
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None

async def call_gemini(prompt, data, model="models/gemini-2.5-flash-preview-04-17"):
    """Call Gemini with the given prompt and data."""
    combined_input = f"{prompt}\n\nDATA:\n{json.dumps(data, indent=2)}"
    logger.info(f"Input length: {len(combined_input)} characters")
    
    try:
        model = genai.GenerativeModel(model)
        response_text = ""
        
        # Use streaming to get faster response
        for chunk in model.generate_content(combined_input, stream=True):
            if chunk.text:
                response_text += chunk.text
        
        return response_text
    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return None

async def generate_report_card(analysis_data_path, recommendation_data_path, output_path):
    """
    Generate a report card using two LLMs:
    1. First LLM analyzes complexity and accuracy (Parts 1 & 2)
    2. Second LLM provides recommendations (Part 3)
    
    Args:
        analysis_data_path: Path to the analysis data JSON file
        recommendation_data_path: Path to the recommendation data JSON file
        output_path: Path to save the generated report card
    """
    # Get the base directory (mobile-analyzer-service)
    base_dir = Path(__file__).parent.parent
    
    # Define prompt paths
    analysis_prompt_path = base_dir / "analyzer_service" / "prompts" / "report_card_analysis_prompt.txt"
    recommendation_prompt_path = base_dir / "analyzer_service" / "prompts" / "report_card_recommendations_prompt.txt"
    
    # Load prompts
    logger.info(f"Loading analysis prompt from {analysis_prompt_path}")
    analysis_prompt = await load_prompt(analysis_prompt_path)
    if not analysis_prompt:
        return
    
    logger.info(f"Loading recommendation prompt from {recommendation_prompt_path}")
    recommendation_prompt = await load_prompt(recommendation_prompt_path)
    if not recommendation_prompt:
        return
    
    # Load data
    logger.info(f"Loading analysis data from {analysis_data_path}")
    analysis_data = await load_data(analysis_data_path)
    if not analysis_data:
        return
    
    logger.info(f"Loading recommendation data from {recommendation_data_path}")
    recommendation_data = await load_data(recommendation_data_path)
    if not recommendation_data:
        return
    
    # Call both LLMs concurrently
    logger.info("Calling both LLMs concurrently")
    total_start_time = time.time()
    
    # Create tasks for both LLM calls
    analysis_task = call_gemini(analysis_prompt, analysis_data)
    recommendation_task = call_gemini(recommendation_prompt, recommendation_data)
    
    # Run both tasks concurrently
    analysis_result, recommendation_result = await asyncio.gather(analysis_task, recommendation_task)
    
    total_time = time.time() - total_start_time
    logger.info(f"Both LLM calls completed in {total_time:.2f}s")
    
    # Check if either call failed
    if not analysis_result:
        logger.error("Analysis failed")
        return
    
    if not recommendation_result:
        logger.error("Recommendations failed")
        return
    
    # Combine results
    report_card = f"# ENGLISH LEARNER REPORT CARD\n\n"
    report_card += f"## PART 1 & 2: COMPLEXITY AND ACCURACY ANALYSIS\n\n{analysis_result}\n\n"
    report_card += f"## PART 3: GROWTH RECOMMENDATIONS\n\n{recommendation_result}\n\n"
    
    # Write the report card to a file
    logger.info(f"Writing report card to {output_path}")
    try:
        with open(output_path, 'w') as f:
            f.write(report_card)
        logger.info("Report card generation completed successfully")
    except Exception as e:
        logger.error(f"Error writing report card to {output_path}: {e}")

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a report card using LLMs")
    parser.add_argument("analysis_data", help="Path to the analysis data JSON file")
    parser.add_argument("recommendation_data", help="Path to the recommendation data JSON file")
    parser.add_argument("output", help="Path to save the generated report card")
    args = parser.parse_args()
    
    # Generate the report card
    await generate_report_card(args.analysis_data, args.recommendation_data, args.output)

if __name__ == "__main__":
    asyncio.run(main())
