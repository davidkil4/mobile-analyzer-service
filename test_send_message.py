"""
Test script to send messages to a Stream Chat channel.
This simulates messages coming from another user.
"""
import os
import sys
from stream_chat import StreamChat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Stream API credentials from environment variables
STREAM_API_KEY = os.getenv("STREAM_API_KEY")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET")

if not STREAM_API_KEY or not STREAM_API_SECRET:
    print("Error: Stream API credentials not found in environment variables.")
    sys.exit(1)

# Initialize Stream Chat client
client = StreamChat(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)

# Define test user (different from test-user-id to ensure messages appear as from another user)
test_user = {
    "id": "test-other-user",
    "name": "Test Other User",
    "image": "https://getstream.io/random_svg/?id=test-other-user&name=Test+Other+User"
}

# Upsert the user to Stream
client.upsert_user(test_user)

# Get the channel
channel = client.channel("messaging", "default-channel")

# Function to send a message
def send_message(text):
    response = channel.send_message(
        {"text": text},
        user_id=test_user["id"]
    )
    print(f"Message sent: {text}")
    print(f"Response: {response}")
    return response

# Main execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Send the message provided as command line argument
        message_text = " ".join(sys.argv[1:])
        send_message(message_text)
    else:
        # Interactive mode
        print("Enter messages to send (type 'exit' to quit):")
        while True:
            message = input("> ")
            if message.lower() == "exit":
                break
            send_message(message)
