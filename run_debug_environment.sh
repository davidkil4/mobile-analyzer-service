#!/bin/bash
# Script to run the Analyzer Service with enhanced logging for debugging duplication issues

echo "=== Starting Analyzer Service Debug Environment ==="
echo "This script will run the Analyzer Service with enhanced logging."

# Define paths
ANALYZER_PATH="/Users/davidkil/projects/analyzer_service mobile"

# Function to check if a port is already in use
port_in_use() {
  lsof -i:$1 >/dev/null 2>&1
  return $?
}

# Check if port is available
if port_in_use 8001; then
  echo "Error: Port 8001 is already in use. Please stop any services using this port."
  exit 1
fi

# Set environment variables for more verbose logging
export LOG_LEVEL=DEBUG
export PYTHONPATH=$PYTHONPATH:$ANALYZER_PATH

# Create log directory
mkdir -p logs

# Start Analyzer Service with direct console output
echo "Starting Analyzer Service with enhanced logging on port 8001..."
cd "$ANALYZER_PATH"

# Print instructions for running the frontend
echo ""
echo "=== IMPORTANT: Run these commands in separate terminals ==="
echo "1. For the Stream Token Service (in a new terminal):"
echo "   cd \"/Users/davidkil/projects/goodtalk expo/goodtalk\" && npm start"
echo ""
echo "2. For the frontend app (in another terminal):"
echo "   cd \"/Users/davidkil/projects/goodtalk expo/goodtalk\" && npx expo start"
echo ""
echo "The Analyzer Service will start now with debug logging enabled."
echo "Press Ctrl+C to stop the Analyzer Service."
echo "=================================================="
echo ""

# Run the Analyzer Service in the foreground with direct console output
python -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --log-level debug
