import requests
import time
import json
import os

API_URL = "http://127.0.0.1:8000/message"
CONVERSATION_ID = "real-conv-600-son"

# Load real utterances from the 600_son_converted.json file
INPUT_FILE = "input_files/600_son_converted.json"

print(f"Loading real utterances from {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)
    utterances = data.get("utterances", [])

# Filter to only include student utterances (not interviewer)
student_utterances = [u for u in utterances if u.get("speaker").lower() == "student"]
print(f"Loaded {len(utterances)} total utterances, {len(student_utterances)} from student speaker.")

# Send only student utterances for analysis
start_time = time.time()
batch_times = []

for i, utterance in enumerate(student_utterances):
    # Create request data using real utterance information
    data = {
        "conversation_id": CONVERSATION_ID,
        "user_id": utterance.get("id", f"user-{i}"),
        "speaker": utterance.get("speaker", "student"),  # Always student in this filtered set
        "text": utterance.get("text", ""),
        "timestamp": utterance.get("timestamp", int(time.time()))
    }
    
    # Send the utterance to the API
    batch_start = time.time()
    response = requests.post(API_URL, json=data)
    resp_json = response.json()
    
    # Print batch analysis information
    if resp_json.get("batch_analyzed"):
        batch_time = resp_json.get("analysis_time_seconds", 0)
        batch_times.append(batch_time)
        print(f"Batch analyzed at message {i+1}: batch_id={resp_json.get('batch_id')}")
        print(f"  Analysis time: {batch_time:.2f}s")
        print(f"  Analysis results: {len(resp_json.get('analysis_results', []))} utterances")

total_send_time = time.time() - start_time
print(f"\nSent {len(utterances)} utterances in {total_send_time:.2f}s")
if batch_times:
    print(f"Average batch analysis time: {sum(batch_times)/len(batch_times):.2f}s")

# Trigger end conversation
print(f"\nNow triggering end_conversation for any remaining messages...")
END_URL = f"http://127.0.0.1:8000/end_conversation/{CONVERSATION_ID}"
end_start_time = time.time()
end_response = requests.post(END_URL)
end_json = end_response.json()
end_time = time.time() - end_start_time

print("End conversation response:")
print(f"  Final batch analyzed: {len(end_json.get('final_analysis_results', []))} utterances")
print(f"  End conversation processing time: {end_time:.2f}s")

# Print clustering result information
clustering_result = end_json.get('clustering_result', {})
print(f"\nClustering results:")
print(f"  Status: {clustering_result.get('status')}")
print(f"  Total analyzed: {clustering_result.get('total_analyzed')} utterances")
print(f"  Clustering time: {clustering_result.get('time_seconds', 0):.2f}s")

# Print output file paths
output_files = clustering_result.get('output_files', {})
print(f"\nOutput files:")
for file_type, file_path in output_files.items():
    print(f"  {file_type}: {file_path}")
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"    Size: {file_size:.2f} KB")
    else:
        print(f"    File not found!")