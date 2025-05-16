"""
Test script for the Stream Chat Token Service.

This script tests the token generation endpoint by making a request with a mock JWT token.
"""

import requests
import json
import sys

# URL of the Stream Token Service
BASE_URL = "http://localhost:8000"

def test_token_generation():
    """Test the token generation endpoint."""
    # Mock JWT token for testing
    mock_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXItaWQiLCJuYW1lIjoiVGVzdCBVc2VyIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    
    # Make a request to the token generation endpoint
    try:
        response = requests.post(
            f"{BASE_URL}/stream/token",
            headers={"Authorization": f"Bearer {mock_jwt}"}
        )
        
        # Print the response status code
        print(f"Status Code: {response.status_code}")
        
        # Print the response body
        if response.status_code == 200:
            print("Success! Token generated:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error:")
            print(response.text)
        
        return response.status_code == 200
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}")
        print("Make sure the service is running.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Stream Chat Token Service...")
    success = test_token_generation()
    sys.exit(0 if success else 1)
