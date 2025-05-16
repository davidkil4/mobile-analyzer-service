# FastAPI integration for Analyzer Service
# This file adds a REST API layer to your existing batch analysis pipeline.
# It does NOT change your core logic. It simply wraps it for frontend/backend integration.

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import logging
import time
import json
import os
import sys
import subprocess
import asyncio
from pathlib import Path
from glob import glob

# Import database modules
from database.db import init_db
from database.api import router as db_router

# Import test JWT endpoint for authentication testing
from test_jwt_endpoint import setup_user_module_routes

# Import your existing pipeline functions and schemas
from analyzer_service.schemas import InputUtterance, PreprocessedASUnit, MainAnalysisOutput, ContextUtterance
from analyzer_service.preprocessing import run_preprocessing_batch
from analyzer_service.analysis import run_analysis_batch

# Import teaching module generation pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from teaching_module_generation.teaching_main import main_orchestrator

app = FastAPI()
logger = logging.getLogger("api_server")

# Initialize the database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {e}")

# Include the database router
app.include_router(db_router)

# Include the test JWT endpoint router
setup_user_module_routes(app)

# In-memory storage for batching and analysis results (per conversation)
conversations: Dict[str, List[Dict[str, Any]]] = {}
analysis_results: Dict[str, List[Any]] = {}  # Stores all analysis results per conversation

# Track conversation turns (messages grouped by speaker)
conversation_turns: Dict[str, List[List[Dict[str, Any]]]] = {}  # Dict[conversation_id, List[List[message]]]

# Easily configurable batch sizes (change here as needed, or load from env)
BATCH_SIZE = 8  # Decreased batch size to 8 utterances for more frequent analysis
TURNS_BATCH_SIZE = 8  # Number of conversation turns to batch before analysis

# Flag to enable turn-based processing instead of fixed-size batching
USE_TURN_BASED_PROCESSING = True  # Set to False to revert to original fixed-size batching

# Directory for saving clustering results
OUTPUT_DIR = "clustering_output"  # Directory where clustering results are stored by default
project_root = Path(__file__).parent.resolve()  # Get the project root directory

# Define the request schema for incoming messages
class MessageRequest(BaseModel):
    conversation_id: str
    user_id: str
    speaker: str  # e.g., "main user", "other user"
    text: str
    timestamp: Optional[int] = None

class AnalysisResponse(BaseModel):
    batch_id: str
    analysis_results: List[MainAnalysisOutput]

@app.post("/message")
async def receive_message(msg: MessageRequest):
    """
    Receives a message from the frontend and adds it to the conversation batch.
    When enough messages or turns are collected, triggers analysis and returns the results.
    """
    # DEBUG LOGGING: Log detailed information about the incoming message
    logger.info(f"[DEBUG] Received message for conversation {msg.conversation_id}")
    logger.info(f"[DEBUG] Message details: user_id={msg.user_id}, speaker={msg.speaker}, text={msg.text[:30]}...")
    logger.info(f"[DEBUG] Current conversation length: {len(conversations.get(msg.conversation_id, []))}")
    
    # Store the message in the appropriate conversation batch
    conv = conversations.setdefault(msg.conversation_id, [])
    
    # DEBUG LOGGING: Log the conversation state before adding the new message
    logger.info(f"[DEBUG] Conversation {msg.conversation_id} before adding message: {len(conv)} messages")
    
    # Add the message to the conversation
    message_data = msg.dict()
    conv.append(message_data)
    logger.info(f"Received message for conversation {msg.conversation_id}: {msg.text}")
    
    # DEBUG LOGGING: Log the conversation state after adding the new message
    logger.info(f"[DEBUG] Conversation {msg.conversation_id} after adding message: {len(conv)} messages")

    # Default response
    response = {"status": "received", "batch_analyzed": False}

    # If turn-based processing is enabled, track conversation turns
    if USE_TURN_BASED_PROCESSING:
        # Initialize turns for this conversation if not already done
        turns = conversation_turns.setdefault(msg.conversation_id, [])
        
        # Check if this is the first message or a new turn (speaker changed)
        if not turns or (turns[-1] and turns[-1][-1]["speaker"] != msg.speaker):
            # Start a new turn
            turns.append([message_data])
            logger.info(f"[DEBUG] Started new turn for speaker: {msg.speaker}")
        else:
            # Add to the current turn
            turns[-1].append(message_data)
            logger.info(f"[DEBUG] Added message to existing turn for speaker: {msg.speaker}")
        
        # Log the current turn count
        logger.info(f"[DEBUG] Current turn count: {len(turns)}/{TURNS_BATCH_SIZE}")
        
        # If we've collected TURNS_BATCH_SIZE turns, trigger analysis
        if len(turns) >= TURNS_BATCH_SIZE:
            # DEBUG LOGGING: Log turn-based batch processing information
            batch_id = f"turn_batch_{msg.conversation_id}_{len(turns) // TURNS_BATCH_SIZE}"
            logger.info(f"[DEBUG] Processing turn-based batch {batch_id} with {len(turns)} turns")
            
            # Combine all messages in each turn into a single message
            combined_messages = []
            for turn_idx, turn in enumerate(turns):
                # Skip empty turns
                if not turn:
                    continue
                    
                # Get the first message in the turn for metadata
                first_msg = turn[0]
                
                # Combine all texts from this turn with spaces
                combined_text = " ".join([m["text"] for m in turn])
                
                # Create a single message representing the entire turn
                combined_message = {
                    "user_id": first_msg["user_id"],
                    "speaker": first_msg["speaker"],
                    "text": combined_text,
                    "timestamp": first_msg.get("timestamp")
                }
                
                combined_messages.append(combined_message)
            
            # Log total turn count
            logger.info(f"[DEBUG] Processing {len(combined_messages)} turns as analysis units")
            
            # Prepare InputUtterance objects for each combined turn message
            input_utterances = [InputUtterance(**{
                "id": m["user_id"] + "-turn-" + str(idx),
                "speaker": m["speaker"],
                "text": m["text"],
                "timestamp": m.get("timestamp")
            }) for idx, m in enumerate(combined_messages)]
            
            # DEBUG LOGGING: Log the combined turn messages being sent for analysis
            for i, message in enumerate(combined_messages):
                logger.info(f"[DEBUG] Turn batch {batch_id} turn {i+1}: speaker={message['speaker']}, text={message['text'][:30]}...")
            
            # Run preprocessing and analysis with timing
            start_time = time.time()
            logger.info(f"[DEBUG] Starting turn-based batch analysis task for batch {batch_id}")
            preprocessed = run_preprocessing_batch(input_utterances)
            
            # Add context to each preprocessed utterance
            for unit in preprocessed:
                original_utterance_id = unit.original_utterance_id
                # Extract the turn index from the ID (format: user_id-turn-idx)
                turn_parts = original_utterance_id.split('-')
                if len(turn_parts) >= 3 and turn_parts[-2] == "turn":
                    turn_idx = int(turn_parts[-1])
                    if 0 <= turn_idx < len(combined_messages):
                        # Get up to 3 preceding turns as context
                        start_idx = max(0, turn_idx - 3)
                        context_turns = combined_messages[start_idx:turn_idx]
                        # Create ContextUtterance objects
                        unit.context = [ContextUtterance(speaker=m["speaker"], text=m["text"]) 
                                      for m in context_turns]
                    else:
                        # If not found, set empty context
                        unit.context = []
                        logger.warning(f"Could not find original turn {turn_idx} for context")
                else:
                    # If ID format doesn't match expected pattern, set empty context
                    unit.context = []
                    logger.warning(f"Could not parse turn index from utterance ID {original_utterance_id}")
            
            batch_analysis = await run_analysis_batch(preprocessed)
            analysis_time = time.time() - start_time
            
            # Accumulate all batch analysis results for clustering
            all_results = analysis_results.setdefault(msg.conversation_id, [])
            all_results.extend(batch_analysis)
            
            # Reset the turns for this conversation
            conversation_turns[msg.conversation_id] = []
            
            # Prepare response
            batch_id = str(uuid.uuid4())
            response = {
                "status": "analyzed",
                "batch_analyzed": True,
                "batch_id": batch_id,
                "analysis_time_seconds": round(analysis_time, 2),
                "turn_based": True,
                "turns_count": TURNS_BATCH_SIZE,
                "messages_count": len(combined_messages),
                "analysis_results": [ar.dict() if hasattr(ar, 'dict') else ar for ar in batch_analysis]
            }
            logger.info(f"Turn-based batch analyzed for conversation {msg.conversation_id}, batch_id={batch_id}, time={analysis_time:.2f}s")
    
    # If turn-based processing is disabled or we're using the original fixed-size batching
    elif len(conv) % BATCH_SIZE == 0:
        # DEBUG LOGGING: Log batch processing information
        batch_id = f"batch_{msg.conversation_id}_{len(conv) // BATCH_SIZE}"
        logger.info(f"[DEBUG] Processing fixed-size batch {batch_id} with {BATCH_SIZE} messages")
        
        # Prepare InputUtterance objects for the batch
        batch_msgs = conv[-BATCH_SIZE:]
        input_utterances = [InputUtterance(**{
            "id": m["user_id"] + "-" + str(idx),
            "speaker": m["speaker"],
            "text": m["text"],
            "timestamp": m.get("timestamp")
        }) for idx, m in enumerate(batch_msgs)]
        
        # DEBUG LOGGING: Log the messages being sent for analysis
        for i, message in enumerate(batch_msgs):
            logger.info(f"[DEBUG] Batch {batch_id} message {i+1}: speaker={message['speaker']}, text={message['text'][:30]}...")
        
        # Run preprocessing and analysis with timing
        start_time = time.time()
        logger.info(f"[DEBUG] Starting batch analysis task for batch {batch_id}")
        preprocessed = run_preprocessing_batch(input_utterances)
        
        # Add context to each preprocessed utterance (similar to main.py)
        for unit in preprocessed:
            original_utterance_id = unit.original_utterance_id
            # Find the original utterance in the conversation
            for i, message in enumerate(conv):
                # Check if this is the message that generated this AS unit
                # The ID format is user_id-index_in_batch
                msg_idx_in_batch = int(original_utterance_id.split('-')[-1])
                batch_start_idx = len(conv) - BATCH_SIZE
                if i == batch_start_idx + msg_idx_in_batch and message["user_id"] in original_utterance_id:
                    # Get up to 3 preceding messages as context
                    start_idx = max(0, i - 3)
                    context_messages = conv[start_idx:i]
                    # Create ContextUtterance objects
                    unit.context = [ContextUtterance(speaker=m["speaker"], text=m["text"]) 
                                   for m in context_messages]
                    break
            else:
                # If not found, set empty context
                unit.context = []
                logger.warning(f"Could not find original utterance {original_utterance_id} for context")
        
        batch_analysis = await run_analysis_batch(preprocessed)
        analysis_time = time.time() - start_time
        
        # Accumulate all batch analysis results for clustering
        all_results = analysis_results.setdefault(msg.conversation_id, [])
        all_results.extend(batch_analysis)
        batch_id = str(uuid.uuid4())
        response = {
            "status": "analyzed",
            "batch_analyzed": True,
            "batch_id": batch_id,
            "analysis_time_seconds": round(analysis_time, 2),
            "turn_based": False,
            "analysis_results": [ar.dict() if hasattr(ar, 'dict') else ar for ar in batch_analysis]
        }
        logger.info(f"Fixed-size batch analyzed for conversation {msg.conversation_id}, batch_id={batch_id}, time={analysis_time:.2f}s")
    
    return response

@app.post("/end_conversation/{conversation_id}")
async def end_conversation(conversation_id: str, background_tasks: BackgroundTasks):
    """
    Endpoint to be called when the user presses the "Analyze Conversation" button in the frontend.
    Analyzes any remaining messages in the batch, triggers clustering, and starts teaching module generation.
    
    FRONTEND INTEGRATION GUIDE - "ANALYZE CONVERSATION" BUTTON:
    
    1. WHEN TO CALL THIS ENDPOINT:
       - This endpoint should be called when the user clicks the "Analyze Conversation" button
       - The button should only be enabled when there are messages in the conversation
       - Typically shown at the end of a chat session or when the user wants to get learning insights
    
    2. USER EXPERIENCE FLOW:
       - User chats with the system, sending multiple messages
       - When ready for analysis, user clicks "Analyze Conversation" button
       - Frontend shows a loading indicator and calls this endpoint
       - As soon as clustering results return, show them to the user
       - As teaching modules become available, show them incrementally
    
    3. IMPLEMENTATION REQUIREMENTS:
       - The conversation_id must match the ID used when sending messages
       - This is a POST request with no body required
       - Handle the response in stages (clustering first, then modules)
       - Implement proper error handling and loading states
       - Consider adding a timeout (30-60 seconds) for very long conversations
    
    4. RESPONSE HANDLING:
       - This endpoint returns IMMEDIATELY after clustering completes
       - Teaching module generation continues in the background
       - The response includes clustering results and a "teaching_modules.status" of "generation_started"
       - Frontend should implement a polling mechanism to check for new modules
       - Use the /teaching_modules/list/{conversation_id} endpoint to poll for available modules
       - Recommended polling interval: Start with 3 seconds, then increase to 5-10 seconds
       - Continue polling until you have enough modules or a reasonable timeout (60-120 seconds)
    
    Example frontend implementation:
    ```javascript
    // Button click handler for "Analyze Conversation"
    async function handleAnalyzeConversation() {
      // 1. Update UI to show loading state
      setIsAnalyzing(true);
      setAnalysisStatus('Analyzing conversation...');
      
      try {
        // 2. Call end_conversation endpoint
        const endResponse = await fetch(`/end_conversation/${conversationId}`, { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        
        if (!endResponse.ok) {
          throw new Error(`Analysis failed: ${endResponse.status}`);
        }
        
        const clusteringData = await endResponse.json();
        
        // 3. Display clustering results immediately
        setAnalysisStatus('Analysis complete! Loading teaching modules...');
        displayClusteringResults(clusteringData);
        
        // 4. Start polling for teaching modules
        let moduleCount = 0;
        let attempts = 0;
        
        const pollForModules = async () => {
          try {
            const modulesResponse = await fetch(`/teaching_modules/list/${conversationId}`);
            if (!modulesResponse.ok) {
              throw new Error(`Failed to fetch modules: ${modulesResponse.status}`);
            }
            
            const modulesData = await modulesResponse.json();
            
            // If we have new modules, display them
            if (modulesData.count > moduleCount) {
              setAnalysisStatus(`Found ${modulesData.count} teaching modules`);
              const newModules = modulesData.modules.slice(moduleCount);
              await fetchAndDisplayModules(newModules);
              moduleCount = modulesData.count;
            }
            
            // Continue polling with exponential backoff
            attempts++;
            if (attempts < 20 && moduleCount < 10) { // Adjust these thresholds as needed
              const delay = Math.min(3000 + attempts * 1000, 10000); // Start at 3s, max 10s
              setTimeout(pollForModules, delay);
            } else {
              // Polling complete
              setIsAnalyzing(false);
              setAnalysisStatus(moduleCount > 0 ? 
                `Analysis complete! Found ${moduleCount} teaching modules.` : 
                'Analysis complete, but no teaching modules were generated.');
            }
          } catch (error) {
            console.error('Error polling for modules:', error);
            setIsAnalyzing(false);
            setAnalysisStatus('Error loading teaching modules. Please try again.');
          }
        };
        
        // Start polling
        pollForModules();
        
      } catch (error) {
        console.error('Error analyzing conversation:', error);
        setIsAnalyzing(false);
        setAnalysisStatus('Error analyzing conversation. Please try again.');
      }
    }
    
    // In your component's JSX:
    // <Button 
    //   onClick={handleAnalyzeConversation} 
    //   disabled={!hasMessages || isAnalyzing}
    // >
    //   {isAnalyzing ? 'Analyzing...' : 'Analyze Conversation'}
    // </Button>
    // <div>{analysisStatus}</div>
    ```
    """
    conv = conversations.get(conversation_id, [])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or already processed.")
    
    # DEBUG LOGGING: Log conversation state at end_conversation call
    logger.info(f"[DEBUG] end_conversation called for conversation {conversation_id}")
    logger.info(f"[DEBUG] Total messages in conversation: {len(conv)}")
    
    remaining = len(conv) % BATCH_SIZE
    logger.info(f"[DEBUG] Remaining messages to analyze: {remaining}")
    
    # DEBUG LOGGING: Log all messages in the conversation to check for duplicates
    logger.info(f"[DEBUG] All messages in conversation {conversation_id}:")
    for i, message in enumerate(conv):
        logger.info(f"[DEBUG] Message {i+1}: speaker={message['speaker']}, text={message['text'][:30]}...")
    
    final_batch_results = []
    # Process any remaining messages in the conversation
    if USE_TURN_BASED_PROCESSING:
        # Process any remaining turns
        turns = conversation_turns.get(conversation_id, [])
        if turns:
            remaining_turns = len(turns)
            logger.info(f"Processing {remaining_turns} remaining turns for conversation {conversation_id}")
            
            # Combine all messages in each turn into a single message
            combined_messages = []
            for turn_idx, turn in enumerate(turns):
                # Skip empty turns
                if not turn:
                    continue
                    
                # Get the first message in the turn for metadata
                first_msg = turn[0]
                
                # Combine all texts from this turn with spaces
                combined_text = " ".join([m["text"] for m in turn])
                
                # Create a single message representing the entire turn
                combined_message = {
                    "user_id": first_msg["user_id"],
                    "speaker": first_msg["speaker"],
                    "text": combined_text,
                    "timestamp": first_msg.get("timestamp")
                }
                
                combined_messages.append(combined_message)
            
            if combined_messages:
                logger.info(f"Processing {len(combined_messages)} combined turns as analysis units")
                
                # Prepare InputUtterance objects for each combined turn message
                input_utterances = [InputUtterance(**{
                    "id": m["user_id"] + "-turn-" + str(idx),
                    "speaker": m["speaker"],
                    "text": m["text"],
                    "timestamp": m.get("timestamp")
                }) for idx, m in enumerate(combined_messages)]
                
                # Run preprocessing and analysis
                preprocessed = run_preprocessing_batch(input_utterances)
                
                # Add context to each preprocessed utterance
                for unit in preprocessed:
                    original_utterance_id = unit.original_utterance_id
                    # Extract the turn index from the ID (format: user_id-turn-idx)
                    turn_parts = original_utterance_id.split('-')
                    if len(turn_parts) >= 3 and turn_parts[-2] == "turn":
                        turn_idx = int(turn_parts[-1])
                        if 0 <= turn_idx < len(combined_messages):
                            # Get up to 3 preceding turns as context
                            start_idx = max(0, turn_idx - 3)
                            context_turns = combined_messages[start_idx:turn_idx]
                            # Create ContextUtterance objects
                            unit.context = [ContextUtterance(speaker=m["speaker"], text=m["text"]) 
                                          for m in context_turns]
                        else:
                            # If not found, set empty context
                            unit.context = []
                            logger.warning(f"Could not find original turn {turn_idx} for context")
                    else:
                        # If ID format doesn't match expected pattern, set empty context
                        unit.context = []
                        logger.warning(f"Could not parse turn index from utterance ID {original_utterance_id}")
                
                # Run analysis on remaining turns
                remaining_analysis = await run_analysis_batch(preprocessed)
                
                # Add to accumulated results
                all_results = analysis_results.setdefault(conversation_id, [])
                all_results.extend(remaining_analysis)
                logger.info(f"Processed {len(combined_messages)} combined turns for conversation {conversation_id}")
                
                # Reset the turns for this conversation
                conversation_turns[conversation_id] = []
    else:
        # Process any remaining messages in the conversation using fixed-size batching
        conv = conversations.get(conversation_id, [])
        remaining = len(conv) % BATCH_SIZE
        
        if remaining > 0:
            logger.info(f"Processing {remaining} remaining messages for conversation {conversation_id}")
            
            # Prepare InputUtterance objects for the remaining messages
            batch_msgs = conv[-remaining:]
            input_utterances = [InputUtterance(**{
                "id": m["user_id"] + "-" + str(idx),
                "speaker": m["speaker"],
                "text": m["text"],
                "timestamp": m.get("timestamp")
            }) for idx, m in enumerate(batch_msgs)]
            
            # Run preprocessing and analysis
            preprocessed = run_preprocessing_batch(input_utterances)
            
            # Add context to each preprocessed utterance
            for unit in preprocessed:
                original_utterance_id = unit.original_utterance_id
                # Find the original utterance in the conversation
                for i, message in enumerate(conv):
                    # Check if this is the message that generated this AS unit
                    msg_idx_in_batch = int(original_utterance_id.split('-')[-1])
                    batch_start_idx = len(conv) - remaining
                    if i == batch_start_idx + msg_idx_in_batch and message["user_id"] in original_utterance_id:
                        # Get up to 3 preceding messages as context
                        start_idx = max(0, i - 3)
                        context_messages = conv[start_idx:i]
                        # Create ContextUtterance objects
                        unit.context = [ContextUtterance(speaker=m["speaker"], text=m["text"]) 
                                       for m in context_messages]
                        break
                else:
                    # If not found, set empty context
                    unit.context = []
                    logger.warning(f"Could not find original utterance {original_utterance_id} for context")
            
            # Run analysis on remaining messages
            remaining_analysis = await run_analysis_batch(preprocessed)
            
            # Add to accumulated results
            all_results = analysis_results.setdefault(conversation_id, [])
            all_results.extend(remaining_analysis)
            logger.info(f"Processed {remaining} remaining messages for conversation {conversation_id}")
    
    # Gather all analysis results for clustering
    all_analyzed = analysis_results.get(conversation_id, [])
    
    # DEBUG LOGGING: Log all analysis results before clustering
    logger.info(f"[DEBUG] Total analysis results for clustering: {len(all_analyzed)}")
    
    # Check for potential duplicates in analysis results
    text_counts = {}
    for i, result in enumerate(all_analyzed):
        if hasattr(result, 'original'):
            text = result.original
        elif isinstance(result, dict) and 'original' in result:
            text = result['original']
        else:
            text = f"unknown-format-{i}"
            
        text_snippet = text[:30]
        text_counts[text_snippet] = text_counts.get(text_snippet, 0) + 1
        
    # Log any potential duplicates
    for text, count in text_counts.items():
        if count > 1:
            logger.warning(f"[DEBUG] Potential duplicate detected: '{text}...' appears {count} times")
    
    # Create output directory if it doesn't exist
    output_dir_path = project_root / OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save all analysis results to a temporary file for clustering
    analysis_file = output_dir_path / f"{conversation_id}_analysis_results.json"
    with open(analysis_file, 'w') as f:
        json.dump(all_analyzed, f)
    
    # --- Run actual clustering analysis ---
    start_time = time.time()
    logger.info(f"Starting clustering analysis on {len(all_analyzed)} utterances")
    
    try:
        # Run the clustering pipeline as a subprocess
        clustering_script_module = "clustering_analysis.run_clustering_pipeline"
        command = [sys.executable, "-m", clustering_script_module, str(analysis_file)]
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        clustering_time = time.time() - start_time
        logger.info(f"Clustering completed in {clustering_time:.2f}s")
        
        # Get paths to output files - clustering pipeline saves to clustering_output directory
        output_base = output_dir_path / analysis_file.stem
        primary_output_path = str(output_dir_path / f"{analysis_file.stem}_primary.json")
        secondary_json_path = str(output_dir_path / f"{analysis_file.stem}_secondary.json")
        prioritized_json_path = str(output_dir_path / f"{analysis_file.stem}_secondary_prioritized.json")
        
        # Create result with file paths
        clustering_result = {
            "status": "clustering_complete",
            "total_analyzed": len(all_analyzed),
            "time_seconds": round(clustering_time, 2),
            "output_files": {
                "analysis_input": str(analysis_file),
                "primary_output": primary_output_path,
                "secondary_output": secondary_json_path,
                "prioritized_output": prioritized_json_path
            }
        }
        
        # Set up the teaching module directory paths for later reference
        teaching_module_dir = os.path.join(
            project_root, 
            "teaching_module_generation", 
            "output_files", 
            "teaching_module_outputs_new", 
            "validated"
        )
        
        # Add teaching module information to the clustering result
        # This will be updated by the background task as modules are generated
        clustering_result["teaching_modules"] = {
            "status": "generation_started",
            "validated_modules_dir": teaching_module_dir,
            "conversation_id": conversation_id
        }
        
        # Start teaching module generation in a background task
        # This allows the API to return immediately while generation continues
        background_tasks.add_task(
            run_teaching_module_generation_background,
            prioritized_json_path=prioritized_json_path,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        clustering_result = {
            "status": "clustering_failed",
            "error": str(e),
            "total_analyzed": len(all_analyzed)
        }
    
    # Clean up memory after clustering
    conversations.pop(conversation_id, None)
    analysis_results.pop(conversation_id, None)
    
    # Generate a batch ID for the final response
    final_batch_id = str(uuid.uuid4())
    
    return {
        "status": "conversation_ended",
        "batch_id": final_batch_id,
        "final_analysis_results": final_batch_results,
        "clustering_result": clustering_result
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring the API server."""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "components": {
            "api": "healthy",
            "database": "healthy"
        }
    }

@app.get("/teaching_module/{filename}")
async def get_teaching_module(filename: str):
    """Retrieve a specific teaching module file by filename.
    
    This endpoint allows the frontend to access the validated teaching modules.
    Only files in the validated directory are accessible.
    
    FRONTEND USAGE GUIDE:
    - Call this endpoint for each module filename returned by /teaching_modules/list/{conversation_id}
    - The response is a complete teaching module JSON object containing:
      - module_id: Unique identifier for the module
      - module_type: Type of teaching module (e.g., CONVERSATIONAL_LOW, DRILL_PRACTICE)
      - source_utterance_info: Original utterance and context information
      - explanations: Teaching explanations (introduction, main, recap)
      - problems: Array of practice problems for this module
    
    Error handling:
    - 404: Module not found (may still be generating or was rejected during validation)
    - 400: Invalid filename (contains path traversal characters)
    - 500: Server error reading the module
    
    Security note: Filenames are sanitized to prevent path traversal attacks
    """
    # Construct the path to the teaching module file
    teaching_module_dir = os.path.join(
        project_root, 
        "teaching_module_generation", 
        "output_files", 
        "teaching_module_outputs_new", 
        "validated"
    )
    
    # Ensure the filename doesn't contain path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(teaching_module_dir, filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Teaching module not found")
    
    # Read and return the file contents
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
        return content
    except Exception as e:
        logger.error(f"Error reading teaching module file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading teaching module: {str(e)}")

async def run_teaching_module_generation_background(prioritized_json_path: str, conversation_id: str):
    """Background task to run the teaching module generation pipeline.
    
    This runs asynchronously after the API has already responded to the client,
    allowing the frontend to start polling for modules immediately.
    
    As each teaching module is validated and saved to the validated directory,
    it becomes immediately available through the /teaching_modules/list/{conversation_id}
    and /teaching_module/{filename} endpoints.
    """
    try:
        logger.info(f"Starting teaching module generation pipeline using {prioritized_json_path}")
        teaching_start_time = time.time()
        
        # Run the teaching module generation pipeline
        await main_orchestrator(input_file=prioritized_json_path)
        
        teaching_time = time.time() - teaching_start_time
        logger.info(f"Teaching module generation completed in {teaching_time:.2f}s")
        
        # The modules are now available in the validated directory
        # The frontend can access them through the /teaching_modules/list/{conversation_id} endpoint
    except Exception as e:
        logger.error(f"Error in background teaching module generation: {str(e)}")

@app.get("/teaching_modules/list/{conversation_id}")
async def list_teaching_modules(conversation_id: str):
    """List all teaching modules available for a specific conversation.
    
    This endpoint returns a list of all validated teaching module filenames for the specified conversation.
    The frontend can use these filenames to request specific modules using the /teaching_module/{filename} endpoint.
    
    FRONTEND POLLING GUIDE:
    - This endpoint should be used to poll for newly generated teaching modules
    - Modules become available incrementally as they are validated
    - The response includes:
      - "modules": List of filenames for available modules
      - "count": Total number of available modules
      - "directory": Directory path where modules are stored (for debugging)
    
    Best practices for polling:
    1. Start polling immediately after receiving the /end_conversation response
    2. Begin with a short interval (3-5 seconds) and gradually increase
    3. Keep track of which modules you've already processed
    4. Continue polling until you have enough modules or reach a timeout
    5. For each new module filename, call /teaching_module/{filename} to get the content
    
    Example response:
    ```json
    {
      "modules": ["module_grammar_0001_abc123.json", "module_vocab_0002_def456.json"],
      "count": 2,
      "directory": "/path/to/validated/modules"
    }
    ```
    """
    # Construct the path to the teaching module directory
    teaching_module_dir = os.path.join(
        project_root, 
        "teaching_module_generation", 
        "output_files", 
        "teaching_module_outputs_new", 
        "validated"
    )
    
    # Check if the directory exists
    if not os.path.exists(teaching_module_dir):
        return {"modules": [], "count": 0}
    
    # Get all JSON files in the validated directory
    module_files = glob(os.path.join(teaching_module_dir, "*.json"))
    
    # Extract just the filenames (without the path)
    module_filenames = [os.path.basename(f) for f in module_files]
    
    return {
        "modules": module_filenames,
        "count": len(module_filenames),
        "directory": teaching_module_dir
    }
