import asyncio
import json
import os
import argparse
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

# Assuming teaching_schemas.py is in the same directory or accessible in PYTHONPATH
from teaching_module_generation.teaching_schemas import TeachingModule # To load the input module

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for API key)
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in environment variables.")
    # Potentially exit or raise an error if the key is essential for the script's core function
else:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Pydantic Schemas for Validation Output ---
class ProblemValidationResult(BaseModel):
    problem_id: str = Field(..., description="The ID of the problem being validated.")
    step1_error_analysis_passes: bool = Field(..., description="True if selected_error analysis is agreed upon and accurate.")
    step1_justification: Optional[str] = Field(None, description="Justification if step1 fails.")
    step2_problem_relevance_passes: bool = Field(..., description="True if the problem directly targets the selected_error.")
    step2_justification: Optional[str] = Field(None, description="Justification if step2 fails.")
    step3_problem_quality_passes: bool = Field(..., description="True if the problem's content and structure are high quality.")
    step3_justification: Optional[str] = Field(None, description="Justification if step3 fails.")
    overall_problem_validation_passes: bool = Field(..., description="True if all three steps pass.")

class ModuleValidationReport(BaseModel):
    module_id: str
    overall_module_validation_status: bool # True if all components (e.g., all problems) pass
    problem_validation_results: List[ProblemValidationResult]
    # We can add explanation_validation_result here later if needed
    validation_errors: List[str] = Field(default_factory=list, description="Errors encountered during the validation process itself.")


# --- LLM Validation Functions --- 
async def call_llm_for_batch_problem_validation(module_source_info: Dict[str, Any], problems_to_validate: List[Dict[str, Any]]) -> Optional[List[ProblemValidationResult]]:
    """
    Calls the LLM to validate multiple problems in a single API call.
    Returns a list of ProblemValidationResult objects, one for each problem.
    """
    if not problems_to_validate:
        return []
    
    logger.info(f"Requesting batch validation for {len(problems_to_validate)} problems")
    
    # Construct a prompt that includes all problems to validate
    problems_json = json.dumps(problems_to_validate, indent=2)
    
    prompt = f"""You are an expert English language teaching material validator.
    
Given the student's original utterance, the corrected version, the specific error of focus, and a list of generated problems, evaluate each problem.

Student and Error Context:
Original Utterance: {module_source_info.get('original')}
Corrected Utterance: {module_source_info.get('corrected')}
Selected Error Type: {module_source_info.get('selected_error', {}).get('type')}
Selected Error Description: {module_source_info.get('selected_error', {}).get('description')}
Selected Error Correction: {module_source_info.get('selected_error', {}).get('correction')}

Problems to Validate:
{problems_json}

Evaluate EACH problem based on the following three criteria and provide your response ONLY in the specified JSON format.

1.  Error Analysis Agreement (step1):
    Based on the 'Student and Error Context', do you agree with the identified 'selected_error' as the primary issue to address?
    Is the 'selected_error' description and correction accurate and relevant?

2.  Problem Relevance (step2):
    Does the problem (its question, options, overall task) directly and effectively target the SAME 'selected_error' identified in step1?
    For Dictation problems, does the 'target_text' effectively highlight or require the student to use the corrected grammatical pattern related to the 'selected_error'?

3.  Problem Quality (step3):
    Is the problem (question, options, feedback, target_text, prompt_text) clear, grammatically correct, and pedagogically sound?
    Are MCQ/FillBlankChoice options plausible distractors?
    Is the feedback accurate and helpful?
    Is the difficulty level appropriate for a student who made the original 'selected_error'?

JSON Output Format - RETURN AN ARRAY OF VALIDATION RESULTS, ONE FOR EACH PROBLEM:
[
  {{
    "problem_id": "problem_id_1",
    "step1_error_analysis_passes": true,
    "step1_justification": null,
    "step2_problem_relevance_passes": true,
    "step2_justification": null,
    "step3_problem_quality_passes": true,
    "step3_justification": null,
    "overall_problem_validation_passes": true
  }}
]

Your response MUST be a valid JSON array containing one object for each problem to validate.
DO NOT include any explanatory text outside the JSON array.
DO NOT use comments in the JSON.
Ensure all boolean values are lowercase (true/false).
"""
    
    # Call the LLM with the constructed prompt
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={"temperature": 0.0, "top_p": 0.95, "top_k": 0},
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        )
        
        llm_output_text = response.text
        
        # Debug the raw output
        logger.debug(f"Raw LLM output: {llm_output_text[:100]}...")
        
        # Try to extract JSON from the response if it's wrapped in markdown code blocks
        if "```json" in llm_output_text and "```" in llm_output_text.split("```json", 1)[1]:
            # Extract content between ```json and the next ```
            json_content = llm_output_text.split("```json", 1)[1].split("```", 1)[0].strip()
            llm_output_text = json_content
        elif "```" in llm_output_text and "```" in llm_output_text.split("```", 1)[1]:
            # Extract content between ``` and the next ```
            json_content = llm_output_text.split("```", 1)[1].split("```", 1)[0].strip()
            llm_output_text = json_content
        
        # If we still don't have valid JSON, try to find array brackets
        if not llm_output_text.strip().startswith("["):
            # Look for the first [ and last ] in the text
            start_idx = llm_output_text.find("[")
            end_idx = llm_output_text.rfind("]")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                llm_output_text = llm_output_text[start_idx:end_idx+1]
        
        # Attempt to parse the JSON
        try:
            validation_results_data = json.loads(llm_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Problematic JSON text: {llm_output_text[:200]}...")
            
            # Fallback: Create default validation results for each problem
            validation_results = []
            for problem in problems_to_validate:
                problem_id = problem.get('problem_id', 'unknown_id')
                logger.warning(f"Creating default failed validation for problem {problem_id} due to JSON parsing error")
                validation_results.append(ProblemValidationResult(
                    problem_id=problem_id,
                    step1_error_analysis_passes=False,
                    step1_justification="Validation failed due to LLM output parsing error",
                    step2_problem_relevance_passes=False,
                    step2_justification="Validation failed due to LLM output parsing error",
                    step3_problem_quality_passes=False,
                    step3_justification="Validation failed due to LLM output parsing error",
                    overall_problem_validation_passes=False
                ))
            return validation_results
        
        # Ensure validation_results_data is a list
        if not isinstance(validation_results_data, list):
            logger.error(f"Expected list but got {type(validation_results_data)}")
            validation_results_data = [validation_results_data] if isinstance(validation_results_data, dict) else []
        
        # Convert each result dict to a ProblemValidationResult object
        validation_results = []
        for result_data in validation_results_data:
            try:
                # Ensure required fields are present
                for field in ['problem_id', 'step1_error_analysis_passes', 'step2_problem_relevance_passes', 'step3_problem_quality_passes']:
                    if field not in result_data:
                        logger.warning(f"Missing required field {field} in validation result. Adding default value.")
                        if field == 'problem_id':
                            # Try to match with one of the input problems
                            for i, problem in enumerate(problems_to_validate):
                                if i == len(validation_results_data) - 1:
                                    result_data['problem_id'] = problem.get('problem_id', f'unknown_problem_{i}')
                                    break
                            else:
                                result_data['problem_id'] = f'unknown_problem_{len(validation_results)}'
                        else:
                            result_data[field] = False
                
                # Ensure overall_problem_validation_passes is consistent
                expected_overall_pass = (
                    result_data.get('step1_error_analysis_passes', False) and 
                    result_data.get('step2_problem_relevance_passes', False) and 
                    result_data.get('step3_problem_quality_passes', False)
                )
                result_data['overall_problem_validation_passes'] = expected_overall_pass
                
                validation_result = ProblemValidationResult(**result_data)
                validation_results.append(validation_result)
            except Exception as e:
                logger.error(f"Failed to parse validation result: {e}")
                logger.error(f"Problematic result data: {result_data}")
                # Continue to the next result
        
        # If we couldn't parse any results, create default failed results
        if not validation_results and problems_to_validate:
            logger.warning("No valid validation results parsed. Creating default failed validations.")
            for i, problem in enumerate(problems_to_validate):
                problem_id = problem.get('problem_id', f'unknown_problem_{i}')
                validation_results.append(ProblemValidationResult(
                    problem_id=problem_id,
                    step1_error_analysis_passes=False,
                    step1_justification="Validation failed due to result parsing error",
                    step2_problem_relevance_passes=False,
                    step2_justification="Validation failed due to result parsing error",
                    step3_problem_quality_passes=False,
                    step3_justification="Validation failed due to result parsing error",
                    overall_problem_validation_passes=False
                ))
        
        return validation_results
    except Exception as e:
        logger.error(f"Error in batch validation LLM call: {e}")
        # Create default validation results for each problem
        validation_results = []
        for i, problem in enumerate(problems_to_validate):
            problem_id = problem.get('problem_id', f'unknown_problem_{i}')
            validation_results.append(ProblemValidationResult(
                problem_id=problem_id,
                step1_error_analysis_passes=False,
                step1_justification=f"Validation failed due to error: {str(e)}",
                step2_problem_relevance_passes=False,
                step2_justification=f"Validation failed due to error: {str(e)}",
                step3_problem_quality_passes=False,
                step3_justification=f"Validation failed due to error: {str(e)}",
                overall_problem_validation_passes=False
            ))
        return validation_results

async def call_llm_for_problem_validation(module_source_info: Dict[str, Any], problem_to_validate: Dict[str, Any]) -> Optional[ProblemValidationResult]:
    """Calls the LLM to validate a single problem based on the 3-step criteria."""
    # This function is kept for backward compatibility
    logger.info(f"Requesting LLM validation for problem_id: {problem_to_validate.get('problem_id')}")
    
    # 1. Construct the detailed prompt using module_source_info and problem_to_validate
    #    This prompt will ask the LLM to evaluate based on the 3 steps and return JSON
    #    matching the ProblemValidationResult schema.
    prompt = f"""You are an expert English language teaching material validator.
Given the student's original utterance, the corrected version, the specific error of focus, and a generated problem, evaluate the problem.

Student and Error Context:
Original Utterance: {module_source_info.get('original')}
Corrected Utterance: {module_source_info.get('corrected')}
Selected Error Type: {module_source_info.get('selected_error', {}).get('type')}
Selected Error Description: {module_source_info.get('selected_error', {}).get('description')}
Selected Error Correction: {module_source_info.get('selected_error', {}).get('correction')}

Problem to Validate:
Type: {problem_to_validate.get('type')}
Problem ID: {problem_to_validate.get('problem_id')}
Question/Template: {problem_to_validate.get('question') or problem_to_validate.get('question_template')}
Options: {problem_to_validate.get('options')}
Feedback: {problem_to_validate.get('feedback')}
Target Text (for Dictation): {problem_to_validate.get('target_text')}

Evaluate the problem based on the following three criteria and provide your response ONLY in the specified JSON format.

1.  Error Analysis Agreement (step1):
    Based on the 'Student and Error Context', do you agree with the identified 'selected_error' as the primary issue to address?
    Is the 'selected_error' description and correction accurate and relevant?

2.  Problem Relevance (step2):
    Does the 'Problem to Validate' (its question, options, overall task) directly and effectively target the SAME 'selected_error' identified in step1?
    For Dictation problems, does the 'target_text' effectively highlight or require the student to use the corrected grammatical pattern related to the 'selected_error'?

3.  Problem Quality (step3):
    Is the 'Problem to Validate' (question, options, feedback, target_text, prompt_text) clear, grammatically correct, and pedagogically sound?
    Are MCQ/FillBlankChoice options plausible distractors?
    Is the feedback accurate and helpful?
    Is the difficulty level appropriate for a student who made the original 'selected_error'?

JSON Output Format:
{{{{
  "problem_id": "{problem_to_validate.get('problem_id')}",
  "step1_error_analysis_passes": boolean,
  "step1_justification": "string_if_step1_false_else_null",
  "step2_problem_relevance_passes": boolean,
  "step2_justification": "string_if_step2_false_else_null",
  "step3_problem_quality_passes": boolean,
  "step3_justification": "string_if_step3_false_else_null",
  "overall_problem_validation_passes": boolean
}}}} 
(Note: 'overall_problem_validation_passes' should be true if and only if step1, step2, and step3_problem_quality_passes are all true.)
"""

    llm_output_text = "" # Initialize to prevent reference before assignment in error cases
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json" # Request JSON output
            ),
            # safety_settings can be configured here if needed, e.g.:
            # safety_settings={ 
            #    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            #    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            #    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            #    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            # }
        )
        response = await model.generate_content_async(prompt)
        llm_output_text = response.text
        logger.debug(f"Raw LLM response for problem {problem_to_validate.get('problem_id')}: {llm_output_text}")

    except Exception as e:
        logger.error(f"Gemini API call failed for problem {problem_to_validate.get('problem_id')}: {e}")
        return None # Indicate failure to get a response

    try:
        # Clean the LLM output if it's wrapped in markdown triple backticks for JSON
        # This might be less necessary if response_mime_type="application/json" works reliably
        if llm_output_text.strip().startswith("```json"):
            llm_output_text = llm_output_text.strip()[7:-3].strip()
        elif llm_output_text.strip().startswith("```"):
            llm_output_text = llm_output_text.strip()[3:-3].strip()
        
        llm_response_data = json.loads(llm_output_text)
        validation_result = ProblemValidationResult(**llm_response_data)
        # Ensure overall_problem_validation_passes is consistent
        expected_overall_pass = (
            validation_result.step1_error_analysis_passes and 
            validation_result.step2_problem_relevance_passes and 
            validation_result.step3_problem_quality_passes
        )
        if validation_result.overall_problem_validation_passes != expected_overall_pass:
            logger.warning(f"LLM overall_problem_validation_passes for {validation_result.problem_id} was inconsistent. Recalculating.")
            validation_result.overall_problem_validation_passes = expected_overall_pass
        return validation_result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response for problem {problem_to_validate.get('problem_id')}: {e}\nRaw response: {llm_output_text}")
        return None
    except Exception as e: # Catch Pydantic validation errors or others
        logger.error(f"Failed to parse LLM response into ProblemValidationResult for problem {problem_to_validate.get('problem_id')}: {e}\nRaw response: {llm_output_text}")
        return None


async def main_validator(module_file_path: str, validated_output_dir: str):
    logger.info(f"Starting validation for module: {module_file_path}")
    
    try:
        with open(module_file_path, 'r', encoding='utf-8') as f:
            module_data = json.load(f)
        teaching_module = TeachingModule(**module_data)
        logger.info(f"Successfully loaded module: {teaching_module.module_id}")
    except FileNotFoundError:
        logger.error(f"Module file not found: {module_file_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from module file {module_file_path}: {e}")
        return
    except Exception as e: # Catches Pydantic validation errors for TeachingModule
        logger.error(f"Error validating module structure for {module_file_path}: {e}")
        return

    problem_validation_results: List[ProblemValidationResult] = []
    valid_problems = []
    invalid_problems = []
    has_valid_problems = False

    # Extract source_utterance_info once, ensure it's a dict for the LLM prompt
    module_source_info = teaching_module.source_utterance_info.model_dump() if teaching_module.source_utterance_info else {}

    if not teaching_module.problems:
        logger.info(f"Module {teaching_module.module_id} has no problems to validate.")
        # Module with no problems is considered valid
        has_valid_problems = True
    else:
        total_problems = len(teaching_module.problems)
        valid_count = 0
        invalid_count = 0
        
        for problem in teaching_module.problems:
            # Ensure problem is a dict for the LLM prompt
            problem_dict = problem.model_dump()
            validation_result = await call_llm_for_problem_validation(module_source_info, problem_dict)
            
            if validation_result:
                problem_validation_results.append(validation_result)
                
                if validation_result.overall_problem_validation_passes:
                    # Problem passed validation - keep it
                    valid_problems.append(problem)
                    valid_count += 1
                    has_valid_problems = True
                    logger.info(f"Problem {problem.problem_id} PASSED validation.")
                else:
                    # Problem failed validation - drop it
                    invalid_problems.append(problem)
                    invalid_count += 1
                    logger.warning(f"Problem {problem.problem_id} FAILED validation - will be dropped.")
            else:
                logger.error(f"Failed to get validation result for problem_id: {problem.problem_id} in module {teaching_module.module_id}")
                invalid_problems.append(problem)
                invalid_count += 1
        
        logger.info(f"Validation summary for {teaching_module.module_id}: {valid_count}/{total_problems} problems passed validation.")
    
    # Create a new module with only the valid problems
    validated_module = teaching_module.model_copy()
    validated_module.problems = valid_problems
    
    # Create and save a validation report
    report = ModuleValidationReport(
        module_id=teaching_module.module_id,
        overall_module_validation_status=has_valid_problems,  # Module passes if it has at least one valid problem
        problem_validation_results=problem_validation_results
    )
    
    # Create directories for validation reports
    os.makedirs(validated_output_dir, exist_ok=True)
    report_filename = f"{teaching_module.module_id}_validation_report.json"
    report_filepath = os.path.join(validated_output_dir, report_filename)
    
    try:
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)
        logger.info(f"Validation report saved to: {report_filepath}")
    except Exception as e:
        logger.error(f"Failed to save validation report {report_filepath}: {e}")

    # Save the module with only valid problems
    if has_valid_problems:
        logger.info(f"Module {teaching_module.module_id} has valid problems - saving with {len(valid_problems)} problems.")
        return validated_module  # Return the validated module for saving
    else:
        logger.warning(f"Module {teaching_module.module_id} has NO valid problems - will be rejected.")
        return None  # Return None to indicate the module should be rejected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a single teaching module JSON file.")
    parser.add_argument("module_file", help="Path to the teaching module JSON file to validate.")
    parser.add_argument("--output_dir", default="./validation_outputs", help="Directory to save validation reports and validated modules.")
    
    args = parser.parse_args()
    
    # Ensure teaching_schemas.py is found if it's not in the same dir
    # This adds the parent directory of the current script to sys.path, 
    # assuming teaching_schemas.py is in the same dir as this script OR 
    # this script is one level down from where teaching_schemas might be (e.g. in a 'scripts' folder)
    # If teaching_schemas is in teaching_module_generation, and this script is also there, this is fine.
    # If this script is in a subfolder of teaching_module_generation, adjust as needed.
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir) # Or script_dir if teaching_schemas is at the same level
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)

    asyncio.run(main_validator(args.module_file, args.output_dir))
