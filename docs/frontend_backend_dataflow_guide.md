Subject: Guide: Frontend to Backend Utterance Data Flow for Analysis Pipeline

Hi [Frontend Developer Name],

This document outlines how utterance data should flow from the frontend to our backend analysis pipeline, clarifies where filtering and context generation happen, and details what the frontend needs to send to ensure the system works as intended.

**Overall Goal:**
The backend analysis pipeline (`main.py`) is designed to analyze **student utterances**. However, to provide accurate and rich context for each student utterance, it now needs to "see" the **full conversation history**, including interviewer utterances.

**1. Frontend Responsibility: Sending All Utterances**

*   **What to Send:** For every message/utterance in the conversation (from both the student and the interviewer), the frontend should send the following information to our FastAPI backend (`api_server.py`):
    *   `id`: A unique identifier for the utterance.
    *   `speaker`: A string indicating who spoke (e.g., `"student"`, `"interviewer"`). This is crucial.
    *   `text`: The actual text content of the utterance.
    *   `conversation_id`: An identifier for the overall conversation session.
    *   Any other relevant metadata (e.g., `timestamp`).
*   **Key Point:** Please ensure that **all** utterances, regardless of speaker, are sent to the backend API. Do **not** filter out interviewer utterances on the frontend. The backend system is now designed to handle this.

**2. Backend API Server (`api_server.py`) Role**

*   **Receiving Data:** The FastAPI server (`api_server.py`) will receive the stream of utterance objects from the frontend.
*   **Batching:** It batches these utterances per `conversation_id` (e.g., every 10 messages or when a conversation ends, as per `docs/deployment_plan.md`).
*   **Invoking the Analysis Pipeline (`main.py`):**
    *   When `api_server.py` is ready to trigger an analysis run (either for a batch or the full conversation), it will prepare an **input JSON file**.
    *   This input JSON file **must contain all the utterances (both student and interviewer)** for that specific batch or conversation segment that it received from the frontend.
    *   It then calls the `main.py` script (our core analysis pipeline), providing the path to this input JSON file.

**3. Core Analysis Pipeline (`main.py`) - How it Works Now**

This is where the primary logic for handling utterances, context, and filtering for processing resides.

*   **Input:** `main.py` receives the path to the input JSON file prepared by `api_server.py`. This file contains all utterances for the segment being analyzed (student and interviewer).

*   **`load_utterances` function in `main.py` (Modified Logic):**
    *   **Reads All Utterances:** This function now reads *every* utterance object from the input JSON.
    *   **Creates Two Internal Lists:**
        1.  `full_conversation_utterances`: A list of all utterances (student and interviewer) parsed from the input JSON. This list is used to build the conversational context.
        2.  `student_utterances_for_processing`: A list containing *only* the student utterances. These are the utterances that will actually go through the main analysis steps (preprocessing, LLM analysis, etc.).
    *   **No More Aggressive Upfront Filtering for Context:** Previously, `load_utterances` would immediately filter out non-student utterances, and only student utterances were available for context. This has changed.

*   **Processing Path:**
    *   Only the `student_utterances_for_processing` are sent through the various analysis chains (translation, segmentation, accuracy analysis, pattern analysis, etc.). So, **we are still only *analyzing* student performance.**

*   **Context Generation (The Key Change):**
    *   For each student utterance being processed, `main.py` now looks at the `full_conversation_utterances` list to find the 3 utterances that immediately preceded it.
    *   This means the `context` field for a student's analyzed unit will now correctly include any interviewer utterances that came just before the student spoke. This provides richer, more accurate context for the analysis models.

**4. Why the Frontend Developer Might Have Had Trouble Finding "The" Filtering Logic:**

*   **Original Filtering:** The *original* primary filtering logic (where interviewer utterances were discarded for all purposes) was indeed in the `load_utterances` function within `main.py`.
*   **Refined Logic:** This logic has been *refined*, not removed. Instead of just discarding interviewer utterances, `main.py` now intelligently separates concerns:
    *   It uses *all* utterances (including interviewer) for building accurate historical context.
    *   It uses *only student* utterances for the actual analysis processing.
*   So, there isn't one single line anymore that says "ignore interviewers." Instead, there's a more nuanced process of creating different views of the data for different purposes (context vs. processing).

**5. What This Means for the Frontend:**

*   **Simplicity:** The frontend's job is straightforward: send all conversation data (student and interviewer turns) for a given `conversation_id` to the backend API (`api_server.py`).
*   **No Frontend Filtering Needed:** The frontend does not need to implement any logic to filter out interviewer utterances before sending them to the backend. Doing so would prevent `main.py` from building the correct context.
*   **Ensure `speaker` Field is Accurate:** The accuracy of the `speaker` field ("student" or "interviewer") is paramount.

**In Summary for the Frontend Developer:**

1.  **Send everything:** Transmit all utterances (student and interviewer) for a conversation to the `/message` endpoint of `api_server.py`.
2.  **Ensure complete data:** Each utterance object sent should include `id`, `speaker` (accurately labeled), `text`, and `conversation_id`.
3.  **Trust the backend:** The backend (`api_server.py` in conjunction with `main.py`) will handle the necessary logic to use interviewer turns for context while only processing student turns for detailed analysis.

This approach ensures that the analysis pipeline has all the information it needs to perform high-quality analysis with proper conversational context.

Let me know if there are any more questions!

Best,
Cascade
