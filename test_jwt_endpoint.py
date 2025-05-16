"""
Simple test endpoint for JWT verification.
This will be used to test the proxy route from the frontend.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Create a router for the user modules endpoints
router = APIRouter(prefix="/user_modules", tags=["user_modules"])

# Security scheme for JWT
security = HTTPBearer()

class ModuleSaveRequest(BaseModel):
    """Request model for saving a teaching module."""
    module_id: str
    problem: str
    explanation: str
    user_id: str
    # Additional fields are allowed
    
    class Config:
        extra = "allow"  # Allow additional fields
        from_attributes = True  # Previously known as orm_mode in Pydantic v1

# Simple mock function to simulate JWT verification
# In production, this would verify the JWT signature and extract claims
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Simple mock JWT verification for testing.
    In production, this would verify the JWT signature using Clerk's public keys.
    """
    try:
        token = credentials.credentials
        # For testing, just extract some basic info from the token
        # In production, you would verify the signature and decode properly
        
        # Just log that we received a token
        print(f"Received token: {token[:10]}...")
        
        # Mock user extraction - in production this would come from verified JWT
        return "test-user-id"
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@router.post("/save")
async def save_module(
    request: Request,
    current_user_id: str = Depends(get_current_user)
):
    """
    Simple echo endpoint that just returns the received JSON data.
    This avoids any ORM/Pydantic model issues for testing.
    """
    # Get the raw JSON data from the request
    try:
        # Parse the raw request body as JSON
        raw_data = await request.json()
        
        # Add test metadata
        response_data = {
            "success": True,
            "message": "Module received successfully",
            "received_data": raw_data,  # Echo back what we received
            "authenticated_user_id": current_user_id,
            "timestamp": datetime.now().isoformat(),
            "test_mode": True
        }
        
        return response_data
    except Exception as e:
        # Return any errors as JSON
        return {
            "success": False,
            "error": f"Failed to process request: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# Function to register these routes with the main app
def setup_user_module_routes(app):
    """Register the user module routes with the main app."""
    app.include_router(router)
