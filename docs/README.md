# Analyzer Service

This repository contains two integrated pipelines for analyzing language learner utterances and generating targeted teaching modules:

1. **Main Analysis Pipeline**: Analyzes utterances, identifies errors, and generates prioritized learning recommendations
2. **Teaching Module Generation Pipeline**: Transforms analysis results into validated teaching modules

Both pipelines are designed to be modular, reproducible, and easy to extend or audit.

# Main Analysis Pipeline

The Analyzer Service provides a robust, multi-stage pipeline for analyzing language learner utterances, identifying errors, and generating prioritized learning recommendations. The pipeline combines preprocessing, analysis, and clustering to deliver comprehensive insights into learner proficiency.

---

## Overview

The pipeline processes raw conversation data through several stages:
1. **Preprocessing**: Translates, segments, and aligns utterances and clauses
2. **Analysis**: Identifies errors, corrects text, analyzes patterns, and calculates scores
3. **Clustering**: Groups similar utterances, defines learning zones, and prioritizes recommendations

**Key Features:**
- Batch processing for efficient handling of large datasets
- LLM-powered analysis using Google's Gemini models
- Hierarchical clustering for proficiency zone identification
- Zone of Proximal Development (ZPD) analysis for targeted learning
- Robust error handling and validation throughout the pipeline
- Seamless integration with the teaching module generation pipeline

---

## Pipeline Components

### 1. **Preprocessing Module**
- **File:** `analyzer_service/preprocessing.py`
- **Key Functions:**
  - Translation of Korean text to English when needed
  - Segmentation of utterances into AS (Analysis of Speech) units
  - Identification of clauses within AS units
  - Alignment of original text with identified clauses
- **What it does:**
  - Prepares raw utterances for detailed linguistic analysis
  - Filters out utterances that are too short for meaningful analysis
  - Ensures proper alignment between original and processed text

---

### 2. **Analysis Module**
- **File:** `analyzer_service/analysis.py`
- **Key Functions:**
  - Correction chain: Fixes grammatical errors in clauses
  - Accuracy chain: Identifies and categorizes errors with severity levels
  - Pattern analysis chain: Detects formulaic language patterns
  - Scoring: Calculates complexity and accuracy scores
- **What it does:**
  - Performs deep linguistic analysis on preprocessed utterances
  - Categorizes errors by type (e.g., Sentence Structure, Word Choice) and severity (critical, moderate, minor)
  - Identifies language patterns that indicate learning progress
  - Generates quantitative scores for complexity and accuracy

---

### 3. **Clustering Analysis Module**
- **Files:** `clustering_analysis/*.py`
- **Key Components:**
  - `ZPD_analyzer.py`: Performs hierarchical clustering and feature extraction
  - `ZPD_regions.py`: Defines learning regions based on tendency zone
  - `ZPD_priority.py`: Prioritizes utterances within each region
- **What it does:**
  - Groups similar utterances based on complexity, accuracy, and error patterns
  - Identifies the learner's "tendency zone" (current proficiency level)
  - Creates learning regions relative to the tendency zone (ZPD/2, ZPD-1, ZPD, ZPD+1)
  - Calculates priority scores to identify the most valuable learning targets

---

### 4. **Pipeline Orchestration**
- **File:** `main.py`
- **How to run:**
  ```bash
  python main.py -i <input_file.json> -o <output_file.json> -b <batch_size>
  ```
- **What it does:**
  - Orchestrates all pipeline stages in sequence
  - Processes data in configurable batches for efficiency
  - Handles file I/O, error reporting, and logging
  - Automatically triggers the clustering analysis pipeline

---

## Running the Pipeline

The pipeline is designed to be run from the command line with the following parameters:

```bash
python main.py -i input_files/your_input.json -o output/your_output.json -b 10
```

Parameters:
- `-i, --input`: Path to the input JSON file containing utterances
- `-o, --output`: Path to the output JSON file for analysis results
- `-b, --batch_size`: Number of utterances to process per batch (default: 10)

Example with logging:
```bash
python main.py -i input_files/660_golf.json -o output/660_golf_analysis.json -b 10 > pipeline_run.log 2>&1
```

---

## Pipeline Output

The pipeline generates several output files:
1. **Primary Analysis** (`your_output.json`): Contains detailed analysis of each utterance
2. **Clustering Results** (`clustering_output/your_output_primary.json`): Contains clustering analysis and tendency zone information
3. **Learning Regions** (`clustering_output/your_output_secondary.json`): Contains utterances organized by learning region
4. **Prioritized Recommendations** (`clustering_output/your_output_secondary_prioritized.json`): Contains prioritized learning recommendations

The prioritized recommendations can be fed directly into the Teaching Module Generation Pipeline for creating targeted learning materials.

---

# Teaching Module Generation Pipeline

This section describes the pipeline for generating validated, LLM-powered English-teaching modules from the analysis results produced by the Main Analysis Pipeline.

---

## Overview

The pipeline transforms prioritized utterance data (from clustering/streaming analysis) into high-quality, validated teaching modules. It leverages large language models (LLMs) for both classification and content generation, and supports both English and Korean utterances.

**Key features:**
- Modular, stage-based processing
- LLM-powered focus selection and explanation/problem generation
- Integrated validation for pedagogical soundness
- Batch and concurrent processing for efficiency
- Clear file hand-offs and output directories

---

## Pipeline Stages

### 1. **Filter & Split**
- **Script:** `filter_and_split.py`
- **Input:** Raw clustering output (e.g., `clustering_output/660_golf_analysis_secondary_prioritized.json`)
- **Output:** 
  - `teaching_input_part1.json` (English, for focus selection)
  - `teaching_input_part2.json` (English, alternate split)
  - `korean_teaching_input.json` (Korean utterances)
- **What it does:**  
  - Filters out low-priority and irrelevant utterances
  - Splits data by language (English/Korean)
  - Applies statistical filtering by zone

---

### 2. **Korean Focus Selection**
- **Script:** `korean_focus_selection.py`
- **Input:** `korean_teaching_input.json`
- **Output:** 
  - `korean_word_focused.json`
  - `korean_phrase_focused.json`
- **What it does:**  
  - Uses an LLM to classify each Korean utterance as either a word or phrase focus
  - Extracts main error information for downstream module generation

---

### 3. **Batch Focus Selection (English)**
- **Script:** `batch_focus_selection.py`
- **Input:** `teaching_input_part1.json`, `teaching_input_part2.json`
- **Output:** 
  - `grammar.json`
  - `patterns.json`
  - `vocabulary_and_small_grammar.json`
- **What it does:**  
  - Uses an LLM to assign a main focus type (`PATTERN`, `GRAMMAR`, `SMALL_GRAMMAR`, `VOCABULARY`) to each utterance
  - Extracts the most relevant error or pattern for module targeting

---

### 4. **Teaching Module Generation & Validation**
- **Script:** `generate_teaching_modules.py`
- **Input:** All categorized files from above
- **Output:** 
  - `teaching_module_outputs_new/validated/` (final modules)
  - `teaching_module_outputs_new/rejected/` (modules that failed validation)
  - `teaching_module_outputs_new/validation_reports/` (detailed LLM validation reports)
- **What it does:**  
  - For each utterance, generates:
    - **Explanations** (using the appropriate LLM prompt)
    - **Problems** (MCQ, fill-in-the-blank, dictation, as appropriate)
  - Runs LLM-powered validation to ensure modules are pedagogically sound and target the correct error
  - Supports concurrent LLM calls for efficiency (notably for Korean phrase modules)

---

### 5. **Pipeline Orchestration**
- **Script:** `teaching_main.py`
- **How to run:**  
  ```bash
  python teaching_main.py -i <path_to_clustering_output.json>
  ```
- **What it does:**  
  - Orchestrates all the above stages in strict sequence
  - Handles file hand-offs, error reporting, directory setup, and logging
  - Can be run as a single command for end-to-end processing

---

## Directory Structure

```
teaching_module_generation/
├── batch_focus_selection.py
├── filter_and_split.py
├── generate_teaching_modules.py
├── korean_focus_selection.py
├── streamlined_validate_module.py
├── teaching_main.py
├── teaching_prompts/
│   ├── explanation_prompt.txt
│   ├── korean_explanation_prompt.txt
│   ├── mcq_revised_prompt.txt
│   ├── fill_blank_prompt.txt
│   └── dictation_prompt.txt
└── output_files/
    ├── output_files/
    └── teaching_module_outputs_new/
```

---

## Environment & Dependencies

- Requires a `.env` file with a valid `GOOGLE_API_KEY` for Gemini LLM calls.
- All dependencies are standard Python packages plus:
  - `langchain`
  - `google-generativeai`
  - `pydantic`
  - `numpy`, `scipy`, `scikit-learn` (for clustering)
  - `matplotlib` (for visualization)
  - `langdetect`
  - `python-dotenv`
- See `requirements.txt` for full details.

---

## Extending or Debugging

- Each stage is modular and can be run/tested independently.
- Logs are written to stdout and can be redirected or captured as needed.
- Intermediate outputs are preserved for easy inspection or re-processing.
- Prompts for LLMs are stored in `teaching_prompts/` for easy editing/tuning.

---

## Example: End-to-End Run

### Running the Analysis Pipeline
```bash
python main.py -i input_files/660_golf.json -o output/660_golf_analysis.json -b 10
```

### Running the Teaching Module Generation Pipeline
```bash
python -m teaching_module_generation.teaching_main -i clustering_output/660_golf_analysis_secondary_prioritized.json
```
*This is the recommended way to run the pipeline for package-based projects, as it ensures all imports work correctly.*

### Complete End-to-End Workflow
```bash
# Run analyzer service pipeline
python main.py -i input_files/660_golf.json -o output/660_golf_analysis.json -b 10

# Generate teaching modules from analysis results
python -m teaching_module_generation.teaching_main -i clustering_output/660_golf_analysis_secondary_prioritized.json
```

- Analysis results will be in `output/` and `clustering_output/`
- Validated teaching modules will be in `output_files/teaching_module_outputs_new/validated/`

---

## Troubleshooting

- If you see missing file errors, ensure all intermediate directories exist and the input file path is correct.
- If LLM calls fail, check your `.env` and API quota.
- Validation reports for rejected modules are in `output_files/teaching_module_outputs_new/validation_reports/`.

---

## Contact

For questions or contributions, please open an issue or pull request!
