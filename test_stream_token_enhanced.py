"""
Enhanced test script for the Stream Chat Token Service.

This script tests the token generation endpoint with either a mock JWT token
or a real Clerk JWT token, depending on availability.
"""

import requests
import json
import sys
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# URL of the Stream Token Service
BASE_URL = "http://localhost:8000"

def test_token_generation(jwt_token=None):
    """
    Test the token generation endpoint.
    
    Args:
        jwt_token: Optional JWT token to use. If not provided, a mock token will be used.
    """
    # If no token is provided, use a mock JWT token for testing
    if not jwt_token:
        # This is a mock JWT token for testing purposes only
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJuYW1lIjoiVGVzdCBVc2VyIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        print("Using mock JWT token (development mode)")
    else:
        print("Using provided JWT token")
    
    # Make a request to the token generation endpoint
    try:
        response = requests.post(
            f"{BASE_URL}/stream/token",
            headers={"Authorization": f"Bearer {jwt_token}"}
        )
        
        # Print the response status code
        print(f"Status Code: {response.status_code}")
        
        # Print the response body
        if response.status_code == 200:
            print("Success! Token generated:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("Error:")
            print(response.text)
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Make sure the service is running.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Stream Chat Token Service')
    parser.add_argument('--token', help='JWT token to use for testing')
    parser.add_argument('--token-file', help='File containing JWT token to use for testing')
    
    args = parser.parse_args()
    
    # Get the JWT token from either the command line argument or a file
    jwt_token = None
    if args.token:
        jwt_token = args.token
    elif args.token_file:
        try:
            with open(args.token_file, 'r') as f:
                jwt_token = f.read().strip()
        except Exception as e:
            print(f"Error reading token file: {e}")
            sys.exit(1)
    
    print("Testing Stream Chat Token Service...")
    success = test_token_generation(jwt_token)
    sys.exit(0 if success else 1)
