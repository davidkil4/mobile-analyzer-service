"""
Stream Chat Token Generation Service

This module provides a FastAPI endpoint for generating Stream Chat tokens
for authenticated users in the GoodTalk mobile application.

It implements proper JWT verification for Clerk tokens and follows an iterative
development approach, starting with a simple implementation and refining it.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from stream_chat import StreamChat
import os
import jwt
import json
import requests
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List, Optional
import time
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import base64
import struct

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream_token_service")

# Get Stream API credentials from environment variables
STREAM_API_KEY = os.getenv("STREAM_API_KEY")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET")

# Get Clerk API credentials from environment variables
CLERK_API_KEY = os.getenv("CLERK_API_KEY")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "clerk.goodtalk.app")
CLERK_JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER", f"https://{CLERK_FRONTEND_API}")

# Initialize Stream client
stream_client = None
try:
    stream_client = StreamChat(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)
    logger.info("Stream Chat client initialized successfully")
    
    # For development mode, create a default channel if it doesn't exist
    if os.getenv("DEV_MODE", "false").lower() == "true":
        try:
            # Check if default channel exists
            test_user_id = "test-user-id"
            channel_id = "default-channel"
            
            # Create channel data
            channel_data = {
                "name": "Default Channel",
                "members": [test_user_id]
            }
            
            # Create the channel using the server client (which has admin permissions)
            channel = stream_client.channel("messaging", channel_id, data=channel_data)
            
            # Create the channel
            response = channel.create(test_user_id)
            
            logger.info(f"Default channel created for test user: {test_user_id}")
            logger.info(f"Channel response: {response}")
        except Exception as e:
            logger.warning(f"Could not create default channel: {e}")
except Exception as e:
    logger.error(f"Failed to initialize Stream Chat client: {e}")
    stream_client = None

# Create a FastAPI app
app = FastAPI(
    title="Stream Chat Token Service",
    description="API for generating Stream Chat tokens for the GoodTalk mobile app",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme for JWT
security = HTTPBearer()

# Cache for JWKs to avoid fetching them for every request
jwks_cache = {
    "keys": [],
    "last_updated": 0,
    "expiry": 3600  # Cache for 1 hour
}

def int_from_base64(value: str) -> int:
    """Decode a base64 string to an integer."""
    decoded = base64.urlsafe_b64decode(value + '=' * (4 - len(value) % 4))
    return int.from_bytes(decoded, byteorder='big')

def get_jwks() -> List[Dict[str, Any]]:
    """
    Fetch JWKs (JSON Web Key Set) from Clerk.
    Uses caching to avoid fetching for every request.
    """
    global jwks_cache
    
    current_time = time.time()
    
    # If cache is valid, return cached keys
    if jwks_cache["keys"] and current_time - jwks_cache["last_updated"] < jwks_cache["expiry"]:
        logger.debug("Using cached JWKs")
        return jwks_cache["keys"]
    
    try:
        # Fetch JWKs from Clerk
        jwks_url = f"{CLERK_JWT_ISSUER}/.well-known/jwks.json"
        logger.info(f"Fetching JWKs from {jwks_url}")
        
        response = requests.get(jwks_url)
        response.raise_for_status()
        
        jwks = response.json()
        
        # Update cache
        jwks_cache["keys"] = jwks.get("keys", [])
        jwks_cache["last_updated"] = current_time
        
        logger.info(f"JWKs fetched successfully, found {len(jwks_cache['keys'])} keys")
        return jwks_cache["keys"]
    except Exception as e:
        logger.error(f"Error fetching JWKs: {e}")
        # If fetching fails but we have cached keys, use them
        if jwks_cache["keys"]:
            logger.warning("Using expired cached JWKs due to fetch error")
            return jwks_cache["keys"]
        raise

def get_public_key(kid: str) -> Optional[str]:
    """
    Get the public key for a specific key ID (kid) from the JWKs.
    """
    jwks = get_jwks()
    
    # Find the key with matching kid
    for key in jwks:
        if key.get("kid") == kid:
            # Convert JWK to PEM format
            if key.get("kty") == "RSA":
                # Extract the modulus and exponent
                e = int_from_base64(key.get("e"))
                n = int_from_base64(key.get("n"))
                
                # Create a public key
                public_numbers = RSAPublicNumbers(e=e, n=n)
                public_key = public_numbers.public_key(default_backend())
                
                # Convert to PEM format
                pem = public_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo
                )
                
                return pem.decode('utf-8')
    
    return None

async def verify_clerk_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify JWT token from Clerk and extract the user ID.
    """
    try:
        token = credentials.credentials
        
        # First, decode the token without verification to get the kid
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        if not kid:
            logger.error("Token header does not contain kid")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing kid in header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get the public key for this kid
        public_key = get_public_key(kid)
        
        if not public_key:
            logger.error(f"Public key not found for kid: {kid}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: unknown key ID",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify and decode the token
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_signature": True},
            audience="goodtalk-app",  # Should match your Clerk application's audience
            issuer=CLERK_JWT_ISSUER,
        )
        
        # Extract user ID from the token
        # Clerk uses 'sub' claim for the user ID
        user_id = decoded.get("sub")
        
        if not user_id:
            logger.error("Token does not contain user ID (sub claim)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.info(f"Successfully verified token for user: {user_id}")
        return user_id
    
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# For development/testing, we provide a fallback to the mock verification
# In production, this should be removed
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Verify JWT token and extract user ID.
    In development mode, falls back to mock verification if Clerk verification fails.
    """
    # Check if we're in development mode
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    
    try:
        # Always try proper verification first
        if CLERK_JWT_ISSUER:
            return await verify_clerk_jwt(credentials)
        
        # If no Clerk issuer is configured and we're in dev mode, use mock verification
        if dev_mode:
            logger.warning("Using mock JWT verification (DEVELOPMENT ONLY)")
            token = credentials.credentials
            logger.info(f"Received token: {token[:10]}...")
            return "test-user-id"
        
        # In production with no Clerk issuer, this is an error
        logger.error("No Clerk JWT issuer configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not properly configured",
        )
    
    except HTTPException as e:
        # In development mode, fall back to mock verification if proper verification fails
        if dev_mode:
            logger.warning(f"JWT verification failed, using mock verification: {e.detail}")
            return "test-user-id"
        # In production, propagate the exception
        raise

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "healthy", "service": "Stream Chat Token Service"}

@app.post("/stream/token")
async def generate_stream_token(
    current_user_id: str = Depends(get_current_user)
):
    """
    Generate a Stream Chat token for the authenticated user.
    The user_id is extracted from the verified JWT token.
    
    This endpoint should be called by the mobile app when initializing Stream Chat.
    """
    if not stream_client:
        logger.error("Stream client not initialized - missing API credentials")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stream Chat service is not properly configured",
        )
    
    try:
        # Generate token for the authenticated user
        token = stream_client.create_token(current_user_id)
        logger.info(f"Generated Stream token for user: {current_user_id}")
        return {"token": token, "user_id": current_user_id}
    except Exception as e:
        logger.error(f"Error generating Stream token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate Stream token: {str(e)}",
        )

@app.post("/stream/channel")
async def create_channel(
    channel_data: dict,
    current_user_id: str = Depends(get_current_user)
):
    """
    Create a Stream Chat channel for the authenticated user.
    This endpoint uses the server-side client which has admin permissions.
    
    Request body should contain:
    - channel_id: Unique identifier for the channel
    - channel_name: Display name for the channel
    """
    if not stream_client:
        logger.error("Stream client not initialized - missing API credentials")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stream Chat service is not properly configured",
        )
    
    try:
        channel_id = channel_data.get("channel_id")
        channel_name = channel_data.get("channel_name")
        
        if not channel_id or not channel_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both channel_id and channel_name are required",
            )
        
        # Create the channel data
        channel_data = {
            "name": channel_name,
            "members": [current_user_id]
        }
        
        # Create the channel using the server client (which has admin permissions)
        channel = stream_client.channel("messaging", channel_id, data=channel_data)
        
        # Create the channel with the user as the creator
        response = channel.create(current_user_id)
        
        # Query the channel to get full details
        channel_response = channel.query()
        
        logger.info(f"Created channel {channel_id} for user {current_user_id}")
        return {
            "channel_id": channel_id, 
            "channel_name": channel_name, 
            "status": "created",
            "channel_data": channel_response
        }
    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create channel: {str(e)}",
        )

@app.get("/stream/channel/{channel_id}")
async def get_channel(
    channel_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    Get a Stream Chat channel for the authenticated user.
    This endpoint uses the server-side client which has admin permissions.
    """
    if not stream_client:
        logger.error("Stream client not initialized - missing API credentials")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stream Chat service is not properly configured",
        )
    
    try:
        # Get the channel using the server client (which has admin permissions)
        channel = stream_client.channel("messaging", channel_id)
        
        # Query the channel to get full details
        response = channel.query()
        
        # Add the user as a member if they're not already
        if current_user_id not in response.get("members", []):
            channel.add_members([current_user_id])
        
        logger.info(f"Retrieved channel {channel_id} for user {current_user_id}")
        return response
    except Exception as e:
        logger.error(f"Error retrieving channel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve channel: {str(e)}",
        )

# Run the app using Uvicorn when this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)