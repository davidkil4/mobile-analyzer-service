import os
import json
from dotenv import load_dotenv
import requests

# Load Gemini API key
def load_api_key():
    load_dotenv()
    # Use GOOGLE_API_KEY for Gemini
    return os.getenv("GOOGLE_API_KEY")

def read_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_output(output_path, data):
    print(f"Attempting to write output to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully executed write command for: {output_path}")
        # Verify file existence immediately after writing
        if os.path.exists(output_path):
            print(f"VERIFIED: File '{output_path}' exists immediately after writing.")
        else:
            print(f"ERROR: File '{output_path}' DOES NOT exist immediately after writing.")
    except Exception as e:
        print(f"Error during file write operation for {output_path}: {e}")

def build_focus_selection_prompt(utterance_list):
    prompt = (
        """
You are an expert English language teacher and curriculum designer specializing in personalized language instruction.

For each utterance, you will see:
- original_text
- corrected_text
- filtering_metadata.reason (a teacher’s summary of the main correction)
- errors_found and pattern_analysis (detailed error and pattern info)

Step-by-step:
1. Read the filtering_metadata.reason and use it as a strong clue for categorization.
2. Double-check the actual texts and error/pattern lists to confirm or override the initial impression.
3. If filtering_metadata.reason and error/pattern analysis suggest different categories, choose the one best supported by the evidence, and explain your reasoning in the 'reason' field.

Output only an array of objects with these keys:
  - utterance_index: <integer>
  - focus_type: "PATTERN"|"GRAMMAR"|"SMALL_GRAMMAR"|"VOCABULARY"
  - reason: a brief explanation of why this type best represents the main difference.
  - selected_error: the single error object from "errors_found" matching the focus_type (include only when focus_type is GRAMMAR, SMALL_GRAMMAR, or VOCABULARY)
  - selected_pattern: the single pattern object from "pattern_analysis" matching the focus_type (include only when focus_type is PATTERN)

Do not output any other fields. Return raw JSON (no fences).

---

Example Inputs and Outputs:

Input:
{
  "utterance_index": 0,
  "original_text": "My computer is note P C err made by N E C.",
  "corrected_text": "My computer is a notebook PC made by NEC.",
  "filtering_metadata": {
    "reason": "The corrections accurately change 'note' to 'notebook' and add the necessary indefinite article, improving clarity and grammatical correctness."
  },
  "errors_found": [...],
  "pattern_analysis": [...]
}
Output:
{
  "utterance_index": 0,
  "focus_type": "VOCABULARY",
  "reason": "The main difference is the correction of 'note' to 'notebook', a word choice issue, as indicated by the filtering reason.",
  "selected_error": { "category": "Word Choice (Lexical)", "severity": "moderate", "error": "Incorrect word choice 'note'", "correction": "notebook" }
}

Input:
{
  "utterance_index": 1,
  "original_text": "Umm. Before, urr twice or three times per month. But this one year, I can't play.",
  "corrected_text": "Before, I played tennis four times a month.",
  "filtering_metadata": {
    "reason": "The correction improves the phrasing and word choice, making the sentence more natural and grammatically sound."
  },
  "errors_found": [...],
  "pattern_analysis": [...]
}
Output:
{
  "utterance_index": 1,
  "focus_type": "PATTERN",
  "reason": "The correction makes the sentence more natural by using a common frame for describing habits, as suggested by the filtering reason.",
  "selected_pattern": { "intention": "Habit Description", "category": "Pattern", "component": "[Action] [frequency] per month", "frequency": 3.0, "context": "Describing habits", "note": null }
}

Input:
{
  "utterance_index": 2,
  "original_text": "She is teacher.",
  "corrected_text": "She is a teacher.",
  "filtering_metadata": {
    "reason": "The correction adds the missing article for grammatical correctness."
  },
  "errors_found": [...],
  "pattern_analysis": [...]
}
Output:
{
  "utterance_index": 2,
  "focus_type": "SMALL_GRAMMAR",
  "reason": "The only difference is the missing article before the noun, which is a minor grammar point.",
  "selected_error": { "category": "Article/Preposition/Determiner/Quantifier", "severity": "minor", "error": "Missing indefinite article before 'notebook'", "correction": "a" }
}

Input:
{
  "utterance_index": 3,
  "original_text": "I am live in London.",
  "corrected_text": "I live in London.",
  "filtering_metadata": {
    "reason": "The correction fixes an incorrect verb tense."
  },
  "errors_found": [...],
  "pattern_analysis": [...]
}
Output:
{
  "utterance_index": 3,
  "focus_type": "GRAMMAR",
  "reason": "The main difference is an incorrect verb tense ('am live' vs. 'live'), which is a significant grammar error.",
  "selected_error": { "type": "Verb Tense/Aspect", "severity": "minor", "description": "Incorrect tense 'am live'", "correction": "live" }
}

---

Now, process the following utterances:
"""
        + json.dumps(utterance_list, ensure_ascii=False, indent=2)
    )
    return prompt

def call_gemini_api(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }
    params = {"key": api_key}
    print(f"-- Calling Gemini API at {url}...")
    response = requests.post(url, headers=headers, params=params, json=data)
    print(f"-- Gemini API response status: {response.status_code}")
    response.raise_for_status()
    try:
        raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"-- Error parsing Gemini response JSON: {e}")
        print(f"-- Raw Response JSON: {response.text}")
        raise RuntimeError("Failed to parse Gemini API response structure.") from e
    
    print("-- Raw text extracted from API response.")
    
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:] 
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3] 
    cleaned_text = cleaned_text.strip() 
    
    print("-- Cleaned text prepared for JSON parsing.")
    return cleaned_text

def merge_focus_with_utterances(original_data, focus_list, mapping_info):
    """Merges the focus information back into the original nested data structure."""
    print("-- Merging focus data back into original structure...")
    if len(focus_list) != len(mapping_info):
        print(f"ERROR: Mismatch between focus list length ({len(focus_list)}) and mapping info length ({len(mapping_info)}). Cannot merge.")
        # Handle this error appropriately - maybe raise an exception or return None
        raise ValueError("Focus list and mapping info length mismatch during merge.")

    for flat_idx, focus_obj in enumerate(focus_list):
        # Check if the utterance_index from LLM matches the expected flat_idx
        # This adds robustness in case the LLM mixes up indices
        llm_idx = focus_obj.get("utterance_index")
        if llm_idx is not None and llm_idx != flat_idx:
            print(f"WARNING: LLM utterance_index {llm_idx} does not match expected flat index {flat_idx}. Using flat index.")
            # Decide how to handle mismatch: use flat_idx (safer) or llm_idx?
            # Sticking with flat_idx for now.

        # Get original location from mapping
        try:
            zone_idx, rec_idx = mapping_info[flat_idx]
            focus_type = focus_obj.get('focus_type')
            reason = focus_obj.get('reason')
            selected_error = focus_obj.get('selected_error')
            selected_pattern = focus_obj.get('selected_pattern')
            original_data['analysis_zones'][zone_idx]['recommendations'][rec_idx]['focus_type'] = focus_type
            original_data['analysis_zones'][zone_idx]['recommendations'][rec_idx]['reason'] = reason
            if selected_error:
                original_data['analysis_zones'][zone_idx]['recommendations'][rec_idx]['selected_error'] = selected_error
            if selected_pattern:
                original_data['analysis_zones'][zone_idx]['recommendations'][rec_idx]['selected_pattern'] = selected_pattern
            
            # Ensure utterance_index is present, using flat_idx from this processing stage.
            # batch_teaching_main.py will re-index later based on its own logic.
            original_data['analysis_zones'][zone_idx]['recommendations'][rec_idx]['utterance_index'] = flat_idx

        except IndexError:
            print(f"ERROR: Invalid flat index {flat_idx} during merge. Max index expected: {len(mapping_info) - 1}")
            # Skip this focus object or raise an error
            continue # Skipping for now
        except KeyError as e:
            print(f"ERROR: Missing key {e} in focus_obj at flat index {flat_idx}: {focus_obj}")
            continue # Skipping for now

    print("-- Focus data merging complete.")
    return original_data

def adjust_focus_type_for_korean(data):
    """Adjusts focus_type for Korean vocabulary errors based on severity.
    
    Rules:
    - Critical Korean vocabulary errors -> PATTERN
    - Moderate Korean vocabulary errors -> GRAMMAR
    - Minor Korean vocabulary errors -> VOCABULARY (no change needed)
    """
    print("\n--- KOREAN VOCABULARY ERROR HANDLING DEBUG ---")
    print(f"Number of zones in data: {len(data.get('analysis_zones', []))}")
    
    korean_errors_found = 0
    adjustments_made = 0
    
    for zone_idx, zone in enumerate(data.get("analysis_zones", [])):
        print(f"Zone {zone_idx}: {zone.get('zone_name')} - {len(zone.get('recommendations', []))} recommendations")
        
        for rec_idx, rec in enumerate(zone.get("recommendations", [])):
            print(f"\nRecommendation {rec_idx}:")
            print(f"  Original text: {rec.get('original', '')[:50]}...")
            print(f"  Current focus_type: {rec.get('focus_type')}")
            
            # Debug selected_error
            selected_error = rec.get("selected_error", {})
            print(f"  Selected error: {selected_error}")
            
            if not selected_error:
                print("  No selected_error found!")
                continue
                
            error_type = selected_error.get("type", "")
            print(f"  Error type: '{error_type}'")
            
            # Check if Korean text is in the original
            has_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in rec.get('original', ''))
            print(f"  Contains Korean characters: {has_korean}")
            
            # Case-insensitive check for Korean vocabulary errors
            if error_type and "korean" in error_type.lower():
                korean_errors_found += 1
                print(f"  FOUND KOREAN ERROR: {error_type}")
                
                severity = selected_error.get("severity", "").lower()
                print(f"  Severity: {severity}")
                
                old_focus_type = rec.get("focus_type")
                
                if severity == "critical":
                    # Change focus type to PATTERN for critical Korean errors
                    rec["focus_type"] = "PATTERN"
                    adjustments_made += 1
                    print(f"  ADJUSTED: {old_focus_type} -> PATTERN for critical Korean vocabulary error")
                elif severity == "moderate":
                    # Change focus type to GRAMMAR for moderate Korean errors
                    rec["focus_type"] = "GRAMMAR"
                    adjustments_made += 1
                    print(f"  ADJUSTED: {old_focus_type} -> GRAMMAR for moderate Korean vocabulary error")
                else:
                    print(f"  NO ADJUSTMENT needed for {severity} severity")
            elif has_korean:
                print(f"  WARNING: Contains Korean text but error type is not 'Korean Vocabulary': {error_type}")
    
    print(f"\nSummary: Found {korean_errors_found} Korean vocabulary errors, made {adjustments_made} adjustments")
    print("--- END KOREAN VOCABULARY ERROR HANDLING DEBUG ---\n")
    return data

def filter_by_types(data, allowed_types):
    """Filters the data to include only recommendations with specified focus_types."""
    filtered_data = {
        "metadata": data.get("metadata", {}),
        "analysis_zones": []
    }
    for zone in data.get("analysis_zones", []):
        filtered_recs = [rec for rec in zone.get("recommendations", []) 
                        if rec.get("focus_type") in allowed_types]
        if filtered_recs:  # Only include zones with at least one matching recommendation
            filtered_data["analysis_zones"].append({
                "zone_name": zone.get("zone_name"),
                "recommendations": filtered_recs
            })
    return filtered_data

def process_focus_selection_for_file(input_path: str, api_key: str):
    """Processes a single input file for focus selection and returns the data with focus types.
    Does not write categorized files; that is handled by the orchestrator.
    """
    # api_key = load_api_key() # API key is now passed as an argument
    if not api_key:
        raise RuntimeError("API_KEY not provided to process_focus_selection_for_file.")
    
    # Load the entire JSON structure
    print(f"-- Reading input file: {input_path}")
    full_data = read_input(input_path)
    print("-- Input file read.")

    # --- Flatten the utterance list ---    
    flat_utterance_list = []
    mapping_info = [] # Stores (zone_idx, rec_idx) tuples
    print("-- Flattening utterance list...")
    try:
        for zone_idx, zone in enumerate(full_data.get('analysis_zones', [])):
            for rec_idx, rec in enumerate(zone.get('recommendations', [])):
                flat_utterance_list.append(rec)
                mapping_info.append((zone_idx, rec_idx))
    except AttributeError as e:
        print(f"ERROR: Input data structure is not as expected (missing 'analysis_zones' or 'recommendations'?): {e}")
        print(f"Data structure received: {type(full_data)}")
        return
    print(f"-- Flattening complete. Found {len(flat_utterance_list)} utterances across {len(full_data.get('analysis_zones', []))} zones.")
    
    if not flat_utterance_list:
        print("No utterances found in the input file. Exiting.")
        # Optionally write the original (empty) structure back?
        # write_output(output_path, full_data) 
        return
        
    # --- Process the flat list --- 
    prompt = build_focus_selection_prompt(flat_utterance_list)
    print("Prompt sent to Gemini. (Truncated):\n", prompt[:1000], "\n---\n")
    llm_response = call_gemini_api(prompt, api_key)

    print(">>> Full Raw LLM Response Received <<<" ) # Keep raw log for debug
    print(llm_response)
    print(">>> End of Raw LLM Response <<<\n")

    # Parse the JSON array returned by Gemini
    try:
        # The cleaned response should now be parsable
        focus_list = json.loads(llm_response)
        print(f"-- Successfully parsed JSON response. Received {len(focus_list)} focus objects.")
    except json.JSONDecodeError as e:
        print(f"\nError: Failed to decode JSON response from LLM even after cleaning.")
        print(f"Error details: {e}")
        print("Cleaned Response Text was:")
        print(llm_response) # Print the text that failed parsing
        return # Exit if parsing fails

    # --- Merge results back into original structure --- 
    try:
        merged_data = merge_focus_with_utterances(full_data, focus_list, mapping_info)
        print(f"-- Focus selection completed. Processed {len(focus_list)} utterances.")
        
        # Apply Korean vocabulary error handling logic
        adjusted_data = adjust_focus_type_for_korean(merged_data)
        print("-- Applied Korean vocabulary error handling adjustments.")
        
        return adjusted_data
    except Exception as e:
        print(f"-- Error during focus selection merge: {e}")
        return None # Return None if merge fails

    # The function now returns the adjusted data before categorization and writing.
    # Categorization and writing will be handled by the calling orchestrator (teaching_main.py)
    # after aggregating results from multiple files.
    # Note: This code is unreachable due to the return statements above


if __name__ == "__main__":
    import argparse
    import concurrent.futures

    parser = argparse.ArgumentParser(description="Batch focus selection for teaching modules.")
    parser.add_argument("--input", nargs='+', required=True, help="Path(s) to input file(s) (space-separated for batch mode)")
    parser.add_argument("--output", nargs='+', required=True, help="Path(s) to output file(s) (space-separated for batch mode)")
    args = parser.parse_args()

    input_files = args.input
    output_files = args.output

    if len(input_files) != len(output_files):
        raise ValueError(f"Number of input files ({len(input_files)}) does not match number of output files ({len(output_files)}) ")

    # Load API key once for all parallel runs if run as standalone
    api_key_main = load_api_key()
    if not api_key_main:
        print("Error: GOOGLE_API_KEY not found in environment. Exiting.")
        exit(1)

    def run_one(inp, out_ignored): # output_path is now ignored by the refactored function
        print(f"Starting batch focus selection for {inp}")
        # The refactored function now only needs input_path and api_key
        processed_data = process_focus_selection_for_file(inp, api_key_main)
        if processed_data:
            print(f"Successfully processed {inp}. Data ready for aggregation by an orchestrator.")
            # For standalone run, we might want to still write the categorized files here
            # For simplicity in this refactoring step, we'll assume an orchestrator handles it.
            # If you need standalone batch_focus_selection.py to write files, 
            # you'd re-add the categorization and writing logic here, operating on 'processed_data'.
            # Example (add this back if needed for standalone functionality):
            # output_dir_standalone = os.path.join(os.path.dirname(inp), "output_files_standalone_bfs") # Define a suitable output dir
            # os.makedirs(output_dir_standalone, exist_ok=True)
            # vocab_s_data = filter_by_types(processed_data, {"VOCABULARY", "SMALL_GRAMMAR"})
            # grammar_s_data = filter_by_types(processed_data, {"GRAMMAR"})
            # patterns_s_data = filter_by_types(processed_data, {"PATTERN"})
            # if vocab_s_data["analysis_zones"]: write_output(os.path.join(output_dir_standalone, "vocabulary_and_small_grammar.json"), vocab_s_data)
            # if grammar_s_data["analysis_zones"]: write_output(os.path.join(output_dir_standalone, "grammar.json"), grammar_s_data)
            # if patterns_s_data["analysis_zones"]: write_output(os.path.join(output_dir_standalone, "patterns.json"), patterns_s_data)
            # print(f"Standalone run: Wrote categorized files to {output_dir_standalone}")
        else:
            print(f"Failed to process {inp}.")
    
    if len(input_files) == 1:
        run_one(input_files[0], output_files[0])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(input_files))) as executor:
            futures = [executor.submit(run_one, inp, out) for inp, out in zip(input_files, output_files)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in parallel batch: {e}")
    print("Batch focus selection finished.")
