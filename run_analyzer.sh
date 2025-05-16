#!/bin/bash

# Run the analyzer service on port 8001
# This allows the Stream token service to continue running on port 8000

echo "Starting Analyzer Service on port 8001..."
uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
