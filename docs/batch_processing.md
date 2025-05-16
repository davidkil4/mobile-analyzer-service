# Analyzer Service 5.0 - Batch Processing Plan

This document outlines the planned workflow for batch processing utterances in the Analyzer Service v5.0.

## Core Requirement:

The primary goal is to process input utterances in batches sequentially through the *entire* analysis pipeline before starting the next batch. This means Batch 1 completes Preprocessing, Analysis, and Teaching generation before Batch 2 begins Preprocessing.

## Workflow Steps:

1.  **Load All Data:**
    *   Read the entire input JSON file (containing a list of `InputUtterance` objects) into memory at the start.
    *   Input file path and batch size will be provided via command-line arguments.

2.  **Create Batches:**
    *   Divide the loaded list of utterances into smaller lists (batches) based on the specified batch size.

3.  **Process Batches Sequentially:**
    *   Iterate through the list of batches one by one.
    *   For **each batch**:
        *   **a. Run Preprocessing:** Pass the current batch of `InputUtterance` objects to a dedicated `run_preprocessing_batch` function. This function will orchestrate all preprocessing steps (translation, segmentation, alignment, clause analysis, etc.).
            *   *Optimization Note:* Within the preprocessing step itself (e.g., translation, correction), aim to use batch LLM calls (like `llm.batch()`) for efficiency where applicable.
        *   **b. Run Analysis:** Pass the preprocessed output for the batch to a `run_analysis_batch` function. This handles scoring (complexity, accuracy) and potentially other analyses.
        *   **c. Run Teaching Generation:** Pass the analysis output for the batch to a `run_teaching_batch` function (to be developed later).
        *   **d. Store Batch Result:** Store the final processed result (including teaching content) for the current batch temporarily (e.g., in a list in memory).

4.  **Aggregate Results:**
    *   After the loop finishes processing all batches, combine the stored results from each individual batch into a single final list or structure.

5.  **Save Output:**
    *   Write the aggregated final results to an output JSON file.

## Implementation Notes:

*   The `main.py` script will orchestrate this workflow.
*   Separate modules/functions will handle the logic for preprocessing, analysis, and teaching generation per batch.
*   Pydantic schemas defined in `schemas.py` will be used for data validation and structure between steps.
