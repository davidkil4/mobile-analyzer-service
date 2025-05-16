"""
Simple script to generate a Clerk test token using the Clerk API.
This is for development purposes only.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Clerk API key from environment variables
CLERK_API_KEY = os.getenv("CLERK_API_KEY")

if not CLERK_API_KEY:
    print("Error: CLERK_API_KEY not found in environment variables")
    exit(1)

# User ID to generate token for
# In a real scenario, this would be a real user ID from your Clerk instance
USER_ID = input("Enter user ID (or press Enter for 'test_user'): ") or "test_user"

# Make a request to Clerk API to create a token
try:
    response = requests.post(
        "https://api.clerk.dev/v1/tokens",
        headers={
            "Authorization": f"Bearer {CLERK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"user_id": USER_ID}
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get("jwt")
        
        if token:
            print("\nGenerated JWT Token:")
            print(token)
            
            # Save token to file
            with open("clerk_jwt_token.txt", "w") as f:
                f.write(token)
            print("\nToken saved to clerk_jwt_token.txt")
            
            print("\nTo test with this token, run:")
            print(f"python test_stream_token_enhanced.py --token-file clerk_jwt_token.txt")
        else:
            print("Error: No token in response")
            print(json.dumps(token_data, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
