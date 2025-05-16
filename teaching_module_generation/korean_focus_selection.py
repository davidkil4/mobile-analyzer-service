import json
import os
import sys
from copy import deepcopy
import requests

# Gemini API call function (adapted from batch_focus_selection.py)
def call_gemini_api(prompt: str, api_key: str):
    """Calls the Gemini API with the provided prompt and API key."""
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
    print(f"-- Calling Gemini API for Korean classification (batch)... Lenght: {len(prompt)}")
    response = requests.post(url, headers=headers, params=params, json=data)
    print(f"-- Gemini API response status: {response.status_code}")
    response.raise_for_status()
    try:
        raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"-- Error parsing Gemini response JSON: {e}")
        print(f"-- Raw Response JSON: {response.text}")
        raise RuntimeError("Failed to parse Gemini API response structure.") from e
    
    print("-- Raw text extracted from API response for Korean classification.")
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:] 
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    print("-- Cleaned text prepared for JSON parsing.")
    return cleaned_text

BATCH_KOREAN_CLASSIFICATION_PROMPT_TEMPLATE = """
You are an expert English language teacher and curriculum designer specializing in bilingual (Korean-English) learner support.

For each utterance, you will see:
- original (the student's utterance, possibly containing Korean and English)
- corrected (the English version, if available)
- errors_found (a list of identified errors, if available)
- pattern_analysis (linguistic patterns, if available)

**Your task:**
1. For each utterance, decide if the main focus is a "KOREAN_WORD" or a "KOREAN_PHRASE".
2. If you decide it is a **KOREAN_PHRASE**:
    - Carefully examine the `pattern_analysis` field to identify the most relevant pattern or sentence structure that the Korean phrase represents.
    - Populate the `selected_error` field using details from `pattern_analysis` (such as `category`, `component`, etc.), describing the pattern, and provide the precise English correction (e.g., a sentence stem or structure).
3. If you decide it is a **KOREAN_WORD**:
    - Identify the specific Korean word(s) that should be replaced with English.
    - Populate the `selected_error` field with:
        - `type`: "Korean Vocabulary"
        - `severity`: "critical"
        - `description`: A short description like "Korean word used instead of English."
        - `correction`: The direct English equivalent word or phrase from the `corrected` field.
4. In all cases, provide a brief `reason` for your classification and error selection.

**Output only an array of objects with these keys:**
- `utterance_index`: <integer>
- `focus_type`: "KOREAN_WORD" or "KOREAN_PHRASE"
- `reason`: a brief explanation of your classification and error selection
- `selected_error`: an object with the following fields:
    - `type`
    - `severity`
    - `description`
    - `correction`

**Do not output any other fields. Output ONLY valid JSON (no markdown, no explanations outside the array). Use double curly braces for all JSON in your output.**

**Example Input:**
[
  {{
    "utterance_index": 0,
    "original": "좀 부자 동네 같아요.",
    "corrected": "It seems like a rather affluent neighborhood.",
    "filtering_metadata": {{
      "reason": "The correction translates the Korean phrase and improves the sentence structure."
    }},
    "errors_found": [
      {{
        "type": "Korean Vocabulary",
        "severity": "critical",
        "description": "Korean phrase used instead of English.",
        "correction": "affluent neighborhood"
      }}
    ],
    "pattern_analysis": [
      {{
        "intention": "Expressing opinion",
        "category": "Sentence_Stem",
        "component": "It seems like [adjective] [noun]",
        "frequency": 2.0,
        "context": "Common way to express an impression.",
        "note": null
      }}
    ]
  }},
  {{
    "utterance_index": 1,
    "original": "책",
    "corrected": "book",
    "filtering_metadata": {{
      "reason": "The correction replaces the Korean word with its English equivalent."
    }},
    "errors_found": [
      {{
        "type": "Korean Vocabulary",
        "severity": "critical",
        "description": "Korean word used instead of English.",
        "correction": "book"
      }}
    ],
    "pattern_analysis": []
  }}
]

**Example Output:**
[
  {{
    "utterance_index": 0,
    "focus_type": "KOREAN_PHRASE",
    "reason": "The main issue is the use of a Korean phrase to express an opinion, as shown in pattern_analysis. The English correction uses the sentence stem 'It seems like ...', matching the pattern.",
    "selected_error": {{
      "type": "Sentence_Stem",
      "severity": "critical",
      "description": "Used Korean phrase for expressing opinion instead of English sentence stem.",
      "correction": "It seems like [adjective] [noun]"
    }}
  }},
  {{
    "utterance_index": 1,
    "focus_type": "KOREAN_WORD",
    "reason": "A single Korean word was used instead of the English equivalent.",
    "selected_error": {{
      "type": "Korean Vocabulary",
      "severity": "critical",
      "description": "Korean word used instead of English.",
      "correction": "book"
    }}
  }}
]

---

Now, process the following utterances:
{batch_korean_texts_json_string}
"""

def build_batch_korean_classification_prompt(utterance_data_list):
    """Prepares the batch prompt for Korean text classification."""
    prompt_input_list = []
    for i, utt_data in enumerate(utterance_data_list):
        # Create a comprehensive input object with all available information
        input_obj = {
            "utterance_index": i,
            "original": utt_data.get('original', '')
        }
        
        # Add corrected text if available
        if 'corrected' in utt_data:
            input_obj['corrected'] = utt_data['corrected']
            
        # Add errors if available
        if 'errors' in utt_data and utt_data['errors']:
            input_obj['errors_found'] = utt_data['errors']
            
        # Add pattern analysis if available
        if 'pattern_analysis' in utt_data and utt_data['pattern_analysis']:
            input_obj['pattern_analysis'] = utt_data['pattern_analysis']
            
        # Add filtering metadata if available
        if 'filtering_metadata' in utt_data:
            input_obj['filtering_metadata'] = utt_data['filtering_metadata']
            
        prompt_input_list.append(input_obj)
    
    batch_json_string = json.dumps(prompt_input_list, ensure_ascii=False, indent=2)
    return BATCH_KOREAN_CLASSIFICATION_PROMPT_TEMPLATE.format(batch_korean_texts_json_string=batch_json_string)

def find_korean_error(errors_list):
    """Searches for a Korean vocabulary error in the list of errors."""
    if not errors_list:
        return None
    for error in errors_list:
        error_type = error.get('type', error.get('category', ''))
        if isinstance(error_type, str) and 'korean' in error_type.lower():
            return {
                "type": error.get('type', error.get('category')),
                "severity": error.get('severity'),
                "description": error.get('description', error.get('message')),
                "correction": error.get('correction')
            }
    return None

def process_korean_focus_selection(input_path: str, output_dir: str, api_key: str):
    """Processes Korean utterances to determine focus type and select errors using a batch API call."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return
    
    # Extract Korean utterances from the nested structure, keeping track of zone_name
    korean_utterances = []  # List of (zone_name, utterance)
    if isinstance(data, dict) and 'analysis_zones' in data:
        # New format with metadata and analysis_zones
        for zone in data.get('analysis_zones', []):
            zone_name = zone.get('zone_name', 'Unknown')
            for utt in zone.get('recommendations', []):
                korean_utterances.append((zone_name, utt))
    elif isinstance(data, list):
        # Old format (direct list of utterances, no zone info)
        for utt in data:
            korean_utterances.append(('Unknown', utt))
    else:
        print(f"Error: Unexpected data format in {input_path}")
        return

    if not korean_utterances:
        print(f"No Korean utterances to process in {input_path}.")
        # Create empty output files if that's the desired behavior
        os.makedirs(output_dir, exist_ok=True)
        output_path_words = os.path.join(output_dir, 'korean_word_focused.json')
        output_path_phrases = os.path.join(output_dir, 'korean_phrase_focused.json')
        with open(output_path_words, 'w', encoding='utf-8') as f: json.dump([], f)
        with open(output_path_phrases, 'w', encoding='utf-8') as f: json.dump([], f)
        print(f"Empty output files created: {output_path_words}, {output_path_phrases}")
        return

    korean_words_focused = []  # List of (zone_name, utt)
    korean_phrases_focused = []  # List of (zone_name, utt)

    print(f"\n-- Starting Korean focus selection for {len(korean_utterances)} utterances from {input_path} using a batch API call.")
    
    # Prepare and make the single API call
    batch_prompt = build_batch_korean_classification_prompt([utt for _, utt in korean_utterances])
    print("\n===== GEMINI BATCH PROMPT SENT =====\n" + batch_prompt[:2000] + ("..." if len(batch_prompt) > 2000 else "") + "\n===== END PROMPT =====\n")
    llm_response_text = call_gemini_api(batch_prompt, api_key)
    print("\n===== RAW GEMINI RESPONSE =====\n" + llm_response_text[:2000] + ("..." if len(llm_response_text) > 2000 else "") + "\n===== END RESPONSE =====\n")
    try:
        llm_classifications = json.loads(llm_response_text)
        if not isinstance(llm_classifications, list):
            raise ValueError("LLM response is not a JSON list.")
        print(f"-- Successfully parsed {len(llm_classifications)} classifications from LLM.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode LLM JSON response for batch classification: {e}")
        print(f"Raw LLM response: {llm_response_text}")
        # Fallback: classify all as KOREAN_PHRASE or handle error appropriately
        # For now, we'll add a fallback classification and continue
        llm_classifications = [
            {"utterance_index": i, "classification": "KOREAN_PHRASE"} 
            for i in range(len(korean_utterances))
        ]
        print("Warning: Using fallback classification (KOREAN_PHRASE for all) due to parsing error.")
    
    # Print each classification object from the LLM to see if it includes selected_error
    print("\n===== DETAILED LLM CLASSIFICATIONS =====\n")
    for i, item in enumerate(llm_classifications):
        print(f"Classification item {i}: {json.dumps(item, ensure_ascii=False, indent=2)}")
        # Check specifically for selected_error
        if 'selected_error' in item:
            print(f"  ✓ Has selected_error field: {json.dumps(item['selected_error'], ensure_ascii=False, indent=2)}")
        else:
            print(f"  ✗ Missing selected_error field")
        # Check for reason
        if 'reason' in item:
            print(f"  ✓ Has reason field: {item['reason']}")
        else:
            print(f"  ✗ Missing reason field")
    print("\n===== END DETAILED CLASSIFICATIONS =====\n")

    # Map LLM classifications back to utterances
    # Store full classification objects from LLM for each utterance index
    classification_map = {}
    llm_response_map = {}
    
    for item in llm_classifications:
        utterance_idx = item.get('utterance_index')
        if utterance_idx is None:
            continue
            
        # Store the full LLM response item for this utterance
        llm_response_map[utterance_idx] = item
        
        # Extract focus_type for backward compatibility
        focus_type = item.get('focus_type')
        if focus_type is None:
            focus_type = item.get('classification')  # Try alternate field name as fallback
            if focus_type:  # If we found it with the alternate name
                print(f"Note: LLM response uses 'classification' instead of 'focus_type' for index {utterance_idx}")
        classification_map[utterance_idx] = focus_type
    
    # Log the classification map for debugging
    print(f"-- Classification map: {classification_map}")
    print(f"-- Have {len(llm_response_map)} detailed classification objects from LLM")

    for i, (zone_name, utt_data) in enumerate(korean_utterances):
        processed_utt = deepcopy(utt_data)
        original_text_snippet = processed_utt.get('original', '')[:50]
        
        # Get the full LLM response for this utterance index
        llm_response = llm_response_map.get(i)
        focus_type_response = classification_map.get(i)
        
        if llm_response and focus_type_response in ["KOREAN_WORD", "KOREAN_PHRASE"]:
            # Use all the rich data from the LLM response
            processed_utt['focus_type'] = focus_type_response
            
            # Use the LLM's custom reason if available
            if 'reason' in llm_response:
                processed_utt['reason'] = llm_response['reason']
            else:
                processed_utt['reason'] = f"Utterance classified as {focus_type_response.lower().replace('_', ' ')} by batch LLM."
                
            # Use the LLM's custom selected_error if available
            if 'selected_error' in llm_response:
                processed_utt['selected_error'] = llm_response['selected_error']
                print(f"Using LLM-provided selected_error for utterance {i}: {processed_utt['selected_error']['description']}")
            else:
                # Fallback to finding a Korean error in the original data
                processed_utt['selected_error'] = find_korean_error(processed_utt.get('errors'))
                print(f"LLM did not provide selected_error for utterance {i}, using fallback error detection")
        else:
            # Fallback if no valid response from LLM
            print(f"Warning: LLM returned unexpected or missing classification for index {i} ('{focus_type_response}'). Utterance: '{original_text_snippet}'. Defaulting to KOREAN_PHRASE.")
            processed_utt['focus_type'] = "KOREAN_PHRASE"
            processed_utt['reason'] = "Utterance classification fallback due to unexpected/missing LLM batch response item."
            processed_utt['selected_error'] = find_korean_error(processed_utt.get('errors'))
            
        # Log a warning if we still don't have a selected_error
        if not processed_utt['selected_error']:
            print(f"Warning: No selected_error found for utterance: {original_text_snippet}")

        # Log if any required fields are missing after processing
        required_fields = ['focus_type', 'reason', 'selected_error']
        for field in required_fields:
            if field not in processed_utt:
                print(f"[LOG] Missing required field '{field}' in processed utterance index {i}: {json.dumps(processed_utt, ensure_ascii=False)[:300]}")

        if processed_utt['focus_type'] == "KOREAN_WORD":
            korean_words_focused.append((zone_name, processed_utt))
        else: # KOREAN_PHRASE or fallback
            korean_phrases_focused.append((zone_name, processed_utt))
        print(f"Processed Korean utterance {i+1}/{len(korean_utterances)}: Focus '{processed_utt['focus_type']}' for '{original_text_snippet}'")

    os.makedirs(output_dir, exist_ok=True)
    output_path_words = os.path.join(output_dir, 'korean_word_focused.json')
    output_path_phrases = os.path.join(output_dir, 'korean_phrase_focused.json')
    
    # Group Korean utterances by their parent zone_name (from input structure)
    korean_words_by_zone = {}
    korean_phrases_by_zone = {}

    for zone_name, utt in korean_words_focused:
        if zone_name not in korean_words_by_zone:
            korean_words_by_zone[zone_name] = []
        korean_words_by_zone[zone_name].append(utt)

    for zone_name, utt in korean_phrases_focused:
        if zone_name not in korean_phrases_by_zone:
            korean_phrases_by_zone[zone_name] = []
        korean_phrases_by_zone[zone_name].append(utt)

    # Create structured output in the same format as batch_focus_selection.py
    # with metadata and analysis_zones, preserving original zone names
    korean_words_output = {
        "metadata": {},
        "analysis_zones": [
            {
                "zone_name": zone_name,
                "recommendations": utterances
            } for zone_name, utterances in korean_words_by_zone.items()
        ]
    }
    
    korean_phrases_output = {
        "metadata": {},
        "analysis_zones": [
            {
                "zone_name": zone_name,
                "recommendations": utterances
            } for zone_name, utterances in korean_phrases_by_zone.items()
        ]
    }

    # Only create the Korean word focused file if there are actually Korean words
    if korean_words_focused:
        with open(output_path_words, 'w', encoding='utf-8') as f_word:
            json.dump(korean_words_output, f_word, indent=2, ensure_ascii=False)
        print(f"-- Korean words focused saved to: {output_path_words} ({len(korean_words_focused)} utterances)")
    else:
        with open(output_path_words, 'w', encoding='utf-8') as f_word:
            json.dump({"metadata": {}, "analysis_zones": []}, f_word, ensure_ascii=False, indent=2)
        print(f"-- No Korean words found. Created empty output file: {output_path_words}")

    # Only create the Korean phrase focused file if there are actually Korean phrases
    if korean_phrases_focused:
        with open(output_path_phrases, 'w', encoding='utf-8') as f_phrase:
            json.dump(korean_phrases_output, f_phrase, indent=2, ensure_ascii=False)
        print(f"-- Korean phrases focused saved to: {output_path_phrases} ({len(korean_phrases_focused)} utterances)")
    else:
        with open(output_path_phrases, 'w', encoding='utf-8') as f_phrase:
            json.dump({"metadata": {}, "analysis_zones": []}, f_phrase, ensure_ascii=False, indent=2)
        print(f"-- No Korean phrases found. Created empty output file: {output_path_phrases}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python korean_focus_selection.py <input_korean_json_path> [api_key]")
        sys.exit(1)
    
    input_korean_json_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else os.getenv('GOOGLE_API_KEY')

    if not api_key:
        print("Error: GOOGLE_API_KEY not found as environment variable or command-line argument.")
        # For robust behavior, you might want to exit if no API key is provided
        # sys.exit(1)
        print("Attempting to proceed without API key (will likely fail if real API call is needed)...")

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, 'output_files')
    
    process_korean_focus_selection(input_korean_json_path, output_dir, api_key)
    print("Korean focus selection process complete.")

if __name__ == '__main__':
    main() # Directly call main for synchronous execution
