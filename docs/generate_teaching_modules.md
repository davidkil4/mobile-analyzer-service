# Plan of Implementation: Enhanced Korean Language Handling (Revised)

This document outlines the revised multi-stage plan to improve the handling of Korean utterances in the teaching module generation pipeline. This approach emphasizes clearer separation of concerns by introducing a dedicated script for Korean focus selection and modifying existing scripts for language segregation and module generation.

---

**Overall Pipeline Workflow:**

1.  **Orchestrator (`teaching_main.py` - Conceptual):** Manages the overall flow, calling the processing scripts in sequence.
2.  **Stage 1: Initial Language Segregation & Filtering ([filter_and_split.py](cci:7://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:0:0-0:0))**
3.  **Stage 2a: Korean-Specific Focus Selection (`korean_focus_selection.py` - New Script)**
4.  **Stage 2b: English/Other Focus Selection (`batch_focus_selection.py` - Modified for this flow)**
5.  **Stage 3: Unified Teaching Module Generation (`generate_teaching_modules.py` - Modified)**

---

## Stage 1: Initial Language Segregation & Filtering ([filter_and_split.py](cci:7://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:0:0-0:0))

**Objective:** Read the main analysis output, filter utterances, detect language, and segregate Korean utterances into a dedicated file, while English/Other utterances proceed through the existing filtering and splitting logic.

**Modifications in [teaching_module_generation/filter_and_split.py](cci:7://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:0:0-0:0):**

1.  **Add `langdetect` Dependency:**
    *   Ensure `langdetect` is installed (`pip install langdetect`) and added to `requirements.txt`.
    *   Import `langdetect` at the top of the script:
        ```python
        from langdetect import detect as detect_language, LangDetectException
        # Ensure an LLM API call utility is available if not already present (e.g., from batch_focus_selection.py or a shared util)
        # from ..utils.llm_api import call_gemini_api # Example if you create a shared util
        ```

2.  **Modify [filter_and_split](cci:1://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:50:0-102:40) function:**
    *   **Integration Point:** After reading the input data (`data = json.load(f)`) and before [zone_statistical_filtering](cci:1://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:9:0-47:25).
    *   **Logic:**
        1.  Iterate through all utterances across all `analysis_zones`.
        2.  For each utterance that passes initial metadata filters (e.g., `filtering_metadata['decision'] == 'KEEP'` if this logic exists or is added here):
            *   Use `langdetect` on `utterance['original']`.
            *   **If Korean:** Append the *entire utterance object* to a new list, `korean_utterances_for_processing`. These utterances will *not* be included in the `zones` list passed to [zone_statistical_filtering](cci:1://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:9:0-47:25) and subsequent A/B splitting.
            *   **If English/Other:** The utterance remains in its zone and proceeds through the existing [zone_statistical_filtering](cci:1://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:9:0-47:25) and A/B splitting logic.
        3.  **Output `korean_teaching_input.json`:** After processing all zones, if `korean_utterances_for_processing` is not empty, write this list to a new file, e.g., `korean_teaching_input.json`. This file will contain an array of full utterance objects.
        4.  The existing logic for `teaching_input_part1.json` and `teaching_input_part2.json` will now only process English/Other utterances.

    *   **Example Snippet (Conceptual within [filter_and_split](cci:1://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:50:0-102:40) function):**
        ```python
        # ... after loading data ...
        all_input_zones = data.get('analysis_zones', [])
        korean_utterances_for_processing = []
        english_other_zones = [] # To rebuild zones for existing processing

        for zone_data in all_input_zones:
            current_zone_kept_english_recs = []
            for rec in zone_data.get('recommendations', []):
                # Assuming an initial metadata filter might exist or be added here
                # if rec.get('filtering_metadata', {}).get('decision', '').upper() != 'KEEP':
                #     continue
                try:
                    lang = detect_language(rec.get('original', ''))
                    if lang == 'ko':
                        korean_utterances_for_processing.append(deepcopy(rec)) # Store a copy
                    else:
                        current_zone_kept_english_recs.append(rec)
                except LangDetectException:
                    current_zone_kept_english_recs.append(rec) # Default to English path on error
            english_other_zones.append({'zone_name': zone_data.get('zone_name'), 'recommendations': current_zone_kept_english_recs})

        # Write Korean utterances to their own file
        if korean_utterances_for_processing:
            korean_output_path = os.path.join(os.path.dirname(output_path_1), 'korean_teaching_input.json') # Ensure output_path_1 is defined
            with open(korean_output_path, 'w', encoding='utf-8') as f_ko:
                json.dump(korean_utterances_for_processing, f_ko, indent=2, ensure_ascii=False)
            print(f"Korean utterances saved to: {korean_output_path}")

        # Original processing continues with only English/Other utterances
        zones_for_english_processing = english_other_zones
        # ... rest of the original filter_and_split logic uses zones_for_english_processing ...
        ```

---

## Stage 2a: Korean-Specific Focus Selection (`korean_focus_selection.py` - New Script)

**Objective:** Process `korean_teaching_input.json`, classify each utterance as "KOREAN_WORD" or "KOREAN_PHRASE" using a simple LLM call, set this as `focus_type`, and populate `selected_error` from pre-existing error data.

**Create `teaching_module_generation/korean_focus_selection.py`:**

1.  **Dependencies:** LLM API call utility, `json`, `os`, `deepcopy`.
2.  **Core Logic:**
    *   Takes `korean_teaching_input.json` as input.
    *   Defines `KOREAN_CLASSIFICATION_PROMPT_TEMPLATE` (as in previous plan iterations).
    *   Iterates through each utterance object from the input file.
    *   For each utterance:
        *   Call LLM with `KOREAN_CLASSIFICATION_PROMPT_TEMPLATE` using `utterance['original']` to get "KOREAN_WORD" or "KOREAN_PHRASE".
        *   Set `utterance['focus_type']` to the LLM response.
        *   Set `utterance['reason']` (e.g., "Utterance identified as Korean word/phrase by LLM.").
        *   **Populate `selected_error`:**
            *   Search `utterance['errors']` (which came from the main analysis pipeline) for an error with `type` (or `category`) indicative of "Korean Vocabulary" (e.g., case-insensitive match for "korean").
            *   If found, `deepcopy` the relevant fields (e.g., `type`, `description`, `correction`, `severity`) from that error object into `utterance['selected_error']`. Ensure the structure matches what `generate_teaching_modules.py` expects for `InputErrorDetail`.
            *   If not found, `selected_error` can be `None` or a default error structure.
    *   Categorize processed utterances into two lists: `korean_words_focused` and `korean_phrases_focused`.
    *   **Output:** `korean_word_focused.json` and `korean_phrase_focused.json`.

---

## Stage 2b: English/Other Focus Selection (`batch_focus_selection.py` - Modified)

**Objective:** Process English/Other utterances from `teaching_input_part1.json` and `teaching_input_part2.json` using its existing complex LLM for error focus determination.

**Modifications in `teaching_module_generation/batch_focus_selection.py`:**

1.  **Input Handling:** Primarily receives English/Other utterances as its main processing task (these are the outputs from Stage 1's English path).
2.  **Core Logic:** The existing `process_focus_selection_for_file` will largely remain for its primary purpose of English error analysis.
3.  **`adjust_focus_type_for_korean` function:** This sub-function should *remain*. It serves as a fallback: if an utterance was somehow misclassified as English by `langdetect` in Stage 1 but the English-focused LLM in `batch_focus_selection.py` still identifies a "Korean Vocabulary" error in it, this function can correctly re-categorize its `focus_type` (e.g., to PATTERN or GRAMMAR based on severity, as per MEMORY[16085d50]).
4.  **Output:** `grammar.json`, `patterns.json`, `vocabulary_and_small_grammar.json` (containing only English/Other focused utterances, unless `adjust_focus_type_for_korean` re-routes some).

---

## Stage 3: Unified Teaching Module Generation (`generate_teaching_modules.py` - Modified)

**Objective:** Generate teaching modules for all focused utterances (both Korean and English/Other) using appropriate prompts.

**Modifications in `teaching_module_generation/generate_teaching_modules.py`:**

1.  **Create New Prompt Files (in `teaching_prompts/`):**
    *   `korean_explanation_prompt.txt` (content as per previous plan: uses `original_utterance` and `corrected_utterance`).
    *   `korean_practice_prompt.txt` (content as per previous plan: uses `corrected_utterance` for dictation).

2.  **Update `generate_teaching_modules.py` Script:**
    *   **Add New Prompts to `PROMPT_PATHS`:** Include `korean_explanation` and `korean_practice`.
    *   **Add New Configurations to `INPUT_FILES_CONFIG`:**
        ```python
        INPUT_FILES_CONFIG = {
            # ... existing grammar, patterns, vocab configs ...
            "korean_word_focused": {
                "file_name": "korean_word_focused.json", # Matches output from korean_focus_selection.py
                "module_id_prefix": "kor_wd_f",
                "default_module_type": "KOREAN_TO_ENGLISH_WORD"
            },
            "korean_phrase_focused": {
                "file_name": "korean_phrase_focused.json", # Matches output from korean_focus_selection.py
                "module_id_prefix": "kor_ph_f",
                "default_module_type": "KOREAN_TO_ENGLISH_PHRASE"
            }
        }
        ```
    *   **Modify `process_single_recommendation` function:**
        *   It will now receive `InputRecommendation` objects where, for Korean inputs, `focus_type` is "KOREAN_WORD" or "KOREAN_PHRASE" (via `source_file_type` derived from the `INPUT_FILES_CONFIG` key) and `selected_error` is already populated (by `korean_focus_selection.py`).
        *   When processing these Korean types:
            *   Use `korean_explanation_prompt.txt`, formatting with `input_rec.original` and `input_rec.corrected`.
            *   Use `korean_practice_prompt.txt`, formatting with `input_rec.corrected`.
            *   The `selected_error` is available if the prompts or logic need to reference it (e.g., for context or severity, though the new prompts are simpler).
            *   The general structure for calling LLM, parsing, and populating `TeachingModule` (as sketched in the previous plan's `process_single_recommendation` modification for Korean types) remains valid.
        *   Ensure the functions `generate_explanation` and `generate_problems` are flexible enough or new specific versions are used for Korean prompts if they are too tied to specific fields not present/relevant for the Korean path. The `selected_error` *will* be present for Korean items due to Stage 2a.

    *   **Schema Considerations (`teaching_schemas.py`):** Remain as per the previous plan (ensure `TeachingModule.module_type` as `str` is fine).

---

## Orchestration (`teaching_main.py` - Conceptual Changes)

*   `teaching_main.py` will need to be updated to call these scripts in the correct order:
    1.  [filter_and_split.py](cci:7://file:///Users/davidkil/projects/analyzer_service%209.0/teaching_module_generation/filter_and_split.py:0:0-0:0) (generates `korean_teaching_input.json`, `teaching_input_part1.json`, `teaching_input_part2.json`).
    2.  `korean_focus_selection.py` (consumes `korean_teaching_input.json`, produces `korean_word_focused.json`, `korean_phrase_focused.json`).
    3.  `batch_focus_selection.py` (consumes `teaching_input_part1.json`, `teaching_input_part2.json`, produces `grammar.json`, etc.).
    4.  `generate_teaching_modules.py` (consumes all focused files from previous steps).

---

This revised plan offers a more modular and potentially robust pipeline for handling Korean and English utterances separately before generating teaching modules.