# Analyzer Service 5.0 - System Architecture

This document describes the high-level architecture for the Analyzer Service v5.0, focusing on achieving sequential batch processing while maintaining a modular and maintainable codebase.

## Core Design Principles:

*   **Sequential Batch Processing:** Each batch of utterances must complete the entire pipeline (Preprocessing -> Analysis -> Teaching) before the next batch begins.
*   **Modularity / Single Responsibility Principle (SRP):** Logic for distinct stages of the pipeline should reside in separate, focused modules.
*   **Clear Orchestration:** A central script should manage the overall workflow, data loading, batching, and the sequential invocation of stage-specific modules.

## Components:

1.  **`main.py` (Root Directory - Orchestrator):**
    *   **Responsibilities:**
        *   Parse command-line arguments (input file path, output file path, batch size).
        *   Load the *entire* set of input utterances from the specified file.
        *   Divide the loaded utterances into batches.
        *   Iterate through the batches sequentially.
        *   **For each batch:**
            *   Call the main processing function from `analyzer_service/preprocessing.py`, passing the current batch data.
            *   Call the main processing function from `analyzer_service/analysis.py`, passing the preprocessed results for the batch.
            *   Call the main processing function from `analyzer_service/teaching.py`, passing the analysis results for the batch.
            *   Store the final, fully processed results for the completed batch.
        *   Aggregate the results from all processed batches.
        *   Save the aggregated results to the specified output file.
    *   **Does NOT contain:** Detailed logic for preprocessing, analysis, or teaching steps.

2.  **`analyzer_service/preprocessing.py` (Module):**
    *   **Responsibilities:**
        *   Contains all functions, LangChain chains, prompts, and logic related *only* to the preprocessing stage (e.g., translation, segmentation, filtering, alignment, clause analysis).
        *   Defines a primary function (e.g., `process_preprocessing_batch`) that accepts a batch of `InputUtterance` objects (or similar initial data).
        *   Performs all preprocessing steps on the batch.
        *   *Internal Optimization:* May use batch LLM calls (e.g., `llm.batch()`) for suitable steps within the preprocessing stage.
        *   Returns the preprocessed data for the entire batch (e.g., a list of `PreprocessedASUnit` objects).

3.  **`analyzer_service/analysis.py` (Module):**
    *   **Responsibilities:**
        *   Contains all logic related *only* to the analysis stage (e.g., complexity scoring, accuracy scoring, error identification, pattern analysis).
        *   Defines a primary function (e.g., `process_analysis_batch`) that accepts a batch of preprocessed data.
        *   Performs all analysis steps on the batch.
        *   Returns the analysis results for the batch (e.g., a list of `MainAnalysisOutput` objects).

4.  **`analyzer_service/teaching.py` (Module):**
    *   **Responsibilities:**
        *   Contains all logic related *only* to generating teaching content based on the analysis.
        *   Defines a primary function (e.g., `process_teaching_batch`) that accepts a batch of analysis results.
        *   Generates teaching content for the batch.
        *   Returns the teaching content along with potentially merged final results for the batch.

5.  **`analyzer_service/schemas.py` (Module):**
    *   Defines Pydantic schemas for data structures passed between stages (e.g., `InputUtterance`, `PreprocessedASUnit`, `MainAnalysisOutput`, `FinalOutput`). Ensures data consistency.

6.  **`analyzer_service/prompts/` (Directory):**
    *   Stores all prompt `.txt` files used by the different stages.

7.  **Configuration Files (Root Directory):**
    *   `logging.conf`: Configures logging.
    *   `.env` (Gitignored): Stores sensitive information like API keys.
    *   `requirements.txt`: Lists project dependencies.

## Data Flow (Per Batch):

`main.py` -> `preprocessing.py` -> `main.py` -> `analysis.py` -> `main.py` -> `teaching.py` -> `main.py` (stores result)

This cycle repeats for each batch before moving to the next.

## Benefits:

*   Achieves the required sequential end-to-end batch processing.
*   Promotes code organization and maintainability through modularity.
*   Simplifies testing by allowing stages to be tested more independently.
*   Avoids overly large, monolithic script files.
