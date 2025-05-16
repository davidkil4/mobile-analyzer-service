#!/bin/bash
# Script to run the analyzer service with enhanced logging for debugging duplication issues

echo "Starting analyzer service with enhanced logging..."

# Set environment variables for more verbose logging
export LOG_LEVEL=DEBUG
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the analyzer service on port 8001
echo "Running analyzer service on port 8001..."
python -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --log-level debug

# Note: Run the frontend in a separate terminal with:
# cd /Users/davidkil/projects/goodtalk\ expo/goodtalk
# npx expo start
