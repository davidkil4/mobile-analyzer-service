# Deployment Plan

## 1. System Overview

**Architecture:**
- **Frontend (React):** Handles chat UI, interacts with Stream for real-time messaging, and sends user messages to your backend for analysis.
- **Stream:** Third-party service for real-time chat between users.
- **Backend (FastAPI):** Receives messages from the frontend, batches them, runs analysis and clustering pipelines, and returns results or stores them for later retrieval.
- **Analysis Pipelines:**
  - **LangChain Pipeline:** Handles preprocessing and analysis of batches of messages.
  - **Clustering Analysis:** Runs after all messages in a conversation are analyzed.

**Data Flow:**
1. User sends a message in the frontend.
2. Frontend sends the message to both Stream (for chat) and FastAPI (for analysis).
3. FastAPI batches messages per conversation.
4. After every 10 messages, FastAPI triggers the LangChain analysis pipeline.
5. When the conversation ends, any remaining messages are analyzed.
6. Once all messages are analyzed, clustering analysis runs automatically.

---

## 2. Deployment Concerns

### Running FastAPI in Production

- **Use Uvicorn or Gunicorn as the ASGI server.**
  - Example: `uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4`
- **Recommended Cloud Providers:** AWS, Google Cloud, Azure, DigitalOcean, or Heroku.
- **Environment Variables:** Store API keys and secrets in a `.env` file or use your cloud provider’s secret management.

### Scalability

- **FastAPI is highly performant and supports async operations.**
- **Scale up** by increasing the number of workers or running multiple instances behind a load balancer.
- **For high-traffic (10,000+ users):**
  - Use a managed database or Redis for batching and storing messages (not just in-memory).
  - Consider background task processing (Celery, RQ, or FastAPI’s BackgroundTasks) for heavy analysis jobs.

---

## 3. Batching & Analysis Logic

### Message Batching

- Messages are grouped in batches of 10 per conversation.
- Store incoming messages in a dictionary or database keyed by `conversation_id`.
- When 10 messages are collected, trigger the LangChain pipeline for those messages.
- When the conversation ends (user action or timeout), analyze any remaining messages.

### Analysis Pipelines

- **LangChain Pipeline:** Runs preprocessing and analysis on each batch.
- **Clustering Analysis:** After all messages in a conversation are analyzed, run clustering on the full set of results.

### Example Logic (Simplified)

```python
# Pseudocode
conversations = {}

def receive_message(conversation_id, message):
    conversations.setdefault(conversation_id, []).append(message)
    if len(conversations[conversation_id]) % 10 == 0:
        run_langchain_analysis(conversations[conversation_id][-10:])

def end_conversation(conversation_id):
    remaining = len(conversations[conversation_id]) % 10
    if remaining:
        run_langchain_analysis(conversations[conversation_id][-remaining:])
    run_clustering_analysis(conversations[conversation_id])
```

---

## 4. Security & Environment

- **API Keys:** Never hardcode secrets. Use environment variables or secret managers.
- **CORS:** Configure FastAPI to allow requests from your frontend domain.
- **HTTPS:** Always use HTTPS in production.

---

## 5. Monitoring & Maintenance

- **Logging:** Use FastAPI’s logging and set up error tracking (Sentry, LogRocket, etc.).
- **Health Checks:** Implement a `/health` endpoint to monitor server status.
- **Performance Monitoring:** Use tools like Prometheus, Grafana, or your cloud provider’s monitoring.

---

## 6. Next Steps & Recommendations

- **Start with in-memory batching for development.**
- **Move to Redis or a database for production.**
- **Use background tasks for analysis if processing is slow.**
- **Test with simulated traffic before going live.**
- **Document your API endpoints and update this plan as your system evolves.**

---

## 7. Example Deployment Checklist

- [ ] FastAPI server runs and receives messages from frontend.
- [ ] Messages are batched and analyzed correctly.
- [ ] Clustering runs after all analysis is done.
- [ ] Environment variables and secrets are secure.
- [ ] Logging and monitoring are in place.
- [ ] App is tested with simulated user traffic.
- [ ] Ready for cloud deployment and scaling.

---

## 8. Teaching Module Generation & User-Saved Content

### Teaching Module Generation Pipeline

- **Purpose:** After clustering analysis, prioritized utterances are transformed into high-quality, validated English teaching modules (problems, explanations, tips) using a modular, LLM-powered pipeline.
- **Stages:**
  1. **Filter & Split:** Filters and splits utterances by language and priority.
  2. **Focus Selection:** Uses LLMs to classify and focus utterances for teaching (separate for English and Korean).
  3. **Module Generation & Validation:** Generates explanations and problems, validates them for pedagogical soundness, and outputs structured files for frontend consumption.
  4. **Orchestration:** All stages are managed by `teaching_main.py`, which handles file hand-offs, error reporting, and logging.

- **Output Structure:**
  - The pipeline generates individual JSON files for each teaching module (one per utterance)
  - Files are organized into subdirectories:
    - `validated/`: Contains high-quality, validated teaching modules ready for user consumption
    - `rejected/`: Contains modules that failed validation (for debugging/improvement)
    - `validation_reports/`: Contains detailed validation reports for each module
  - **Important:** Only files from the `validated/` subdirectory should be sent to the frontend
  - Each file contains problems, explanations, and other teaching content for a single utterance

- **Data Flow:**
  - Clustering output is fed into the teaching module pipeline.
  - Outputs are validated modules (ready for user review), rejected modules, and validation reports.
  - The frontend displays these modules to users.

### User-Saved Problems & Database Integration

- **Privacy-First Approach:**  
  - **User chat messages and raw analysis data are NOT stored** to protect privacy and comply with regulations.
  - **Only teaching problems, tips, and explanations that a user explicitly chooses to save** are persisted in the database.

- **Database Logic:**
  - When a user clicks "save" on a teaching module in the frontend, an API call is made to the backend.
  - The backend stores the module (problem, explanation, metadata) in a database, associated with the user's ID.
  - Users can later review, update, or delete their saved problems via dedicated API endpoints.

- **Recommended Database:**  
  - Start with SQLite for development (simple, file-based).
  - For production or scaling, use PostgreSQL or another managed relational database.
  - Only teaching modules, not raw chat, are stored.

- **Scalability:**  
  - The database is designed to handle thousands of users and large numbers of saved problems.
  - No sensitive chat data is retained, minimizing privacy risk.

---

## 9. Completed Steps (2025-05-10)

### FastAPI Integration

- **✅ Created `api_server.py`:** 
  - Implemented a FastAPI server that receives messages from the frontend.
  - Added endpoints for `/message` (receives individual messages) and `/end_conversation/{conversation_id}` (processes any remaining messages).
  - Implemented in-memory batching of messages by conversation ID.
  - Integrated with existing preprocessing and analysis pipelines.
  - Added proper async/await handling for asynchronous analysis functions.
  - Included a `/health` endpoint for monitoring.

- **✅ Created `test_send_messages.py`:** 
  - Developed a test script to simulate sending messages to the API.
  - Verified that batching works correctly (every 10 messages triggers analysis).
  - Confirmed that the `/end_conversation` endpoint processes remaining messages.
  - Successfully tested the full message flow from receipt to analysis.

- **✅ Set up a virtual environment:**
  - Created and activated a Python virtual environment for the project.
  - Installed FastAPI, Uvicorn, and other dependencies in the isolated environment.
  - Ensured clean dependency management for development and future deployment.

### Completed Implementation Tasks

- [x] ✅ **Added storage of analysis results for full conversation clustering**
  - Implemented in-memory dictionary to accumulate analysis results across batches
  - Maintained privacy by cleaning up memory after clustering is complete
  - Added proper error handling for edge cases

- [x] ✅ **Implemented clustering trigger in `/end_conversation` endpoint**
  - Added code to run clustering analysis on all accumulated utterances
  - Implemented file-based handoff to the clustering pipeline
  - Added detailed logging and timing metrics

- [x] ✅ **Added context handling for utterances**
  - Implemented tracking of 3 preceding messages as context for each utterance
  - Ensured context information flows through to analysis and clustering outputs
  - Matched the behavior of the main.py pipeline

- [x] ✅ **Enhanced testing with real data**
  - Updated test script to use real conversation data from input files
  - Added filtering to only analyze student messages (not interviewer)  
  - Implemented detailed output reporting for verification

### Completed Implementation Tasks (continued)

- [x] ✅ **Integrated teaching module generation pipeline with API server**
  - Connected the teaching module generation pipeline to run after clustering analysis
  - Added tracking of validated teaching module files
  - Created API endpoints for accessing teaching modules:
    - `/teaching_module/{filename}`: Retrieves a specific teaching module by filename
    - `/teaching_modules/list/{conversation_id}`: Lists all available teaching modules
  - Ensured only validated modules are accessible to the frontend

### Completed Implementation Tasks (continued)

- [x] ✅ **Improved teaching module generation pipeline**
  - Made module generation asynchronous using background tasks
  - Added cleanup for intermediate files to prevent duplicates
  - Enhanced API documentation for frontend developers
  - Implemented polling mechanism for incremental module delivery

- [x] ✅ **Set up database for user-saved teaching modules**
  - Implemented SQLite database with PostgreSQL-compatible design
  - Created User and SavedModule models with SQLAlchemy ORM
  - Added comprehensive CRUD operations for all necessary functions
  - Designed for easy migration to PostgreSQL when needed
  - Created detailed documentation in `docs/database.md`

- [x] ✅ **Created API endpoints for saving and retrieving user-favorite modules**
  - Added `/user_modules/save` endpoint for saving teaching modules
  - Implemented `/user_modules/user/{user_id}` for retrieving user's saved modules
  - Created update and delete endpoints for managing saved modules
  - Added comprehensive frontend integration examples

### Next Implementation Tasks

- [ ] Add authentication and user management
- [ ] Implement frontend components for saved modules
- [ ] Add analytics and usage tracking

### Clustering Integration: Current State

#### 1. Implemented Features

- **Clustering analysis is now triggered when `/end_conversation` is called**
  - The endpoint automatically runs the clustering pipeline on all accumulated utterances
  - Results include file paths to all generated output files
  - Performance metrics (timing) are included in the response

- **Analysis results for all messages are now stored for clustering**
  - Results are accumulated in memory across all batches
  - When `/end_conversation` is called, the full set of results is processed
  - All results are saved to a temporary JSON file for the clustering pipeline

- **Context information is included for each utterance**
  - Each utterance includes up to 3 preceding messages as context
  - Context is properly passed through to the clustering pipeline
  - Output files now include context information for better teaching module generation

#### 2. Performance and Scalability

- **Current Performance:**
  - Average batch analysis time: ~15-20 seconds for 10 utterances
  - Clustering time: ~2 seconds for 30+ utterances
  - Memory usage is minimal and cleaned up after processing

- **Scalability Considerations:**
  - Current in-memory approach works well for thousands of concurrent users
  - For higher volumes, consider Redis or database-backed storage
  - Batch size (currently 10) can be adjusted for performance optimization

---

## 10. Updated Deployment Checklist

- [x] ✅ Teaching module generation pipeline runs after clustering.
- [x] ✅ Validated teaching modules are available for frontend display.
- [x] ✅ Users can save, view, and manage their favorite problems via API.
- [x] ✅ Only user-saved modules are persisted in the database (no raw chat storage).
- [ ] Database is configured and secured for production.
- [ ] All privacy and compliance requirements are met.

---

## 11. Database Implementation (2025-05-11)

### Database Architecture

- **✅ Implemented SQLite for development:**
  - Created a file-based database for storing user-saved teaching modules
  - Designed with a migration path to PostgreSQL for production
  - Used SQLAlchemy ORM for database abstraction

- **✅ Created database models:**
  - `User`: Stores basic user information (ID, username, email)
  - `SavedModule`: Stores user-saved teaching modules with metadata
  - Added relationships between models for efficient queries

- **✅ Implemented CRUD operations:**
  - Create/retrieve users and handle user synchronization
  - Save teaching modules to user's collection
  - Retrieve, update, and delete saved modules
  - Added proper error handling and transaction management

- **✅ Added API endpoints:**
  - POST `/user_modules/save`: Save a teaching module
  - GET `/user_modules/user/{user_id}`: Get all modules saved by a user
  - PUT `/user_modules/{module_id}`: Update a saved module
  - DELETE `/user_modules/{module_id}`: Delete a saved module

- **✅ Created comprehensive documentation:**
  - Added detailed `docs/database.md` with implementation details
  - Included frontend integration guide with code examples
  - Documented migration path to PostgreSQL

### Teaching Module Generation Improvements

- **✅ Made generation asynchronous:**
  - Implemented background tasks for teaching module generation
  - API returns immediately after clustering, while generation continues
  - Added polling mechanism for frontend to retrieve modules as they're created

- **✅ Enhanced frontend integration:**
  - Added detailed documentation for the "Analyze Conversation" button
  - Included JavaScript examples for polling and displaying modules
  - Improved error handling and user feedback

- **✅ Fixed duplicate file issues:**
  - Added cleanup for intermediate files to prevent duplicates
  - Implemented proper directory management for output files
  - Enhanced logging for better debugging

### Next Steps

1. **Authentication Integration:**
   - Connect the User model with the authentication system
   - Implement secure token-based authentication
   - Add user roles and permissions

2. **Frontend Implementation:**
   - Create components for displaying teaching modules
   - Implement save buttons and saved modules page
   - Add user profile and settings

3. **Production Deployment:**
   - Migrate to PostgreSQL for production
   - Set up proper environment variables and secrets
   - Configure CORS and HTTPS

---
