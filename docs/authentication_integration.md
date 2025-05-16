# Authentication Integration Plan

## Overview

This document outlines the authentication integration between the GoodTalk frontend application and the Analyzer Service backend. The plan follows our iterative programming principle: starting with the simplest working solution while ensuring we don't create technical debt.

## Current Authentication System

The GoodTalk frontend uses Clerk for authentication:

- **Authentication Provider**: Clerk
- **Implementation**: Next.js middleware and server-side auth() function
- **Protected Routes**: Chat and stream routes are protected via middleware
- **API Pattern**: Frontend only communicates with Next.js API routes, which handle authentication internally

## Integration Approach

After thorough investigation and security review, we've chosen a JWT-based approach that:

1. Leverages existing Clerk authentication in the frontend
2. Provides strong cryptographic security between services
3. Maintains separation of concerns
4. Follows industry best practices for service-to-service authentication

### Why This Approach?

We chose this approach because:

1. **Security**: It provides cryptographic verification of user identity that doesn't rely on network isolation
2. **Industry Standard**: JWT is the standard approach for secure service-to-service authentication
3. **Self-contained Verification**: The FastAPI backend can independently verify user authentication
4. **Scalability**: This approach scales better as the application grows and network topology becomes more complex

### Alternative Approaches Considered

We also considered a simpler proxy-based approach that would:

- Pass the user ID in a custom header (X-User-ID)
- Trust the Next.js API route based on network isolation
- Require less implementation in the FastAPI backend

While this approach would be simpler to implement initially, it would create significant security technical debt by relying entirely on perfect network configuration to prevent unauthorized access to the FastAPI backend.

## Implementation Details

### 1. Frontend Implementation

Add a new Next.js API route that acts as a proxy to the Analyzer Service, using JWT for authentication:

```typescript
// In /app/api/analyzer/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';

export async function POST(req: NextRequest) {
  // Get authenticated user and token function from Clerk
  const { userId, getToken } = auth();
  
  if (!userId || !getToken) {
    return new NextResponse('Unauthorized', { status: 401 });
  }
  
  try {
    // Get request data
    const data = await req.json();
    
    // Get JWT token from Clerk
    const clerkToken = await getToken();
    if (!clerkToken) {
      return new NextResponse('Unauthorized - Token unavailable', { status: 401 });
    }
    
    // Forward request to analyzer service with JWT token
    const response = await fetch('https://your-analyzer-service.com/api/user_modules/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${clerkToken}`, // Pass JWT token
      },
      body: JSON.stringify(data),
    });
    
    // Return response from analyzer service
    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error calling analyzer service:', error);
    return NextResponse.json(
      { error: 'Failed to communicate with analyzer service' },
      { status: 500 }
    );
  }
}
```

### 2. Backend Implementation

Add JWT verification middleware to the Analyzer Service:

```python
# In api_server.py
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from jose import jwt, JWTError
import requests
from database.crud import get_or_create_user
from database.db import get_db_session

# Security scheme for JWT
security = HTTPBearer()

# Cache for JWKS (JSON Web Key Set)
jwks_cache = None
jwks_url = "https://your-clerk-frontend-api.clerk.accounts.dev/.well-known/jwks.json"

def get_jwks():
    """Get JSON Web Key Set from Clerk."""
    global jwks_cache
    if not jwks_cache:
        response = requests.get(jwks_url)
        jwks_cache = response.json()
    return jwks_cache

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user ID."""
    try:
        token = credentials.credentials
        # Get the key ID from the token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        # Find the matching key in the JWKS
        jwks = get_jwks()
        key = None
        for jwk in jwks.get("keys", []):
            if jwk.get("kid") == kid:
                key = jwk
                break
        
        if not key:
            raise HTTPException(status_code=401, detail="Invalid token: Key not found")
        
        # Verify the token
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience="your-audience",  # Set this to your Clerk application's audience
            issuer="https://clerk.your-domain.com",  # Set this to your Clerk issuer
        )
        
        # Extract user ID from the token (usually in the 'sub' claim)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: User ID not found")
        
        # Create/get user in our database
        with get_db_session() as db:
            user = get_or_create_user(db, user_id)
            return user.user_id
            
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Then use it in endpoints
@app.get("/user_modules/user/{user_id}")
async def get_user_modules(
    user_id: str,
    current_user_id: str = Depends(get_current_user)
):
    # Ensure the user can only access their own modules
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    
    # Continue with the endpoint logic
    # ...
```

## Authentication Flow

1. **User Authentication**:
   - User logs in via Clerk in the GoodTalk frontend
   - Clerk handles authentication and session management

2. **Frontend to Analyzer Service**:
   - Frontend calls the Next.js API route
   - API route verifies the user with Clerk and obtains a JWT token
   - API route forwards the request to the Analyzer Service with the JWT token

3. **Analyzer Service Processing**:
   - Analyzer Service cryptographically verifies the JWT token
   - Extracts the user ID from the verified token
   - Creates/retrieves the user in the database
   - Performs the requested operation
   - Returns the result to the frontend

## Security Considerations

This approach provides strong security through:

1. **Cryptographic Authentication**: JWT tokens are cryptographically signed by Clerk and verified by the Analyzer Service
2. **Independent Verification**: The Analyzer Service can independently verify the authenticity of requests
3. **Authorization**: Users can only access their own data
4. **Secure Communication**: All API calls should use HTTPS
5. **Defense in Depth**: Even if network isolation fails, the JWT verification provides a strong security layer

### Potential Enhancements for Future

If even stronger security is needed in the future, we could:

1. **Add Token Caching**: Implement efficient token caching and validation
2. **Add Rate Limiting**: Prevent abuse through rate limiting
3. **Implement API Keys**: Add API keys for additional service-to-service authentication
4. **Token Revocation**: Implement a token revocation mechanism for immediate session termination

## Testing Plan

To ensure the authentication integration works correctly:

1. **Unit Tests**:
   - Test the JWT verification middleware in isolation
   - Verify that tokens with invalid signatures are rejected
   - Verify that expired tokens are rejected
   - Verify that tokens from unauthorized issuers are rejected

2. **Integration Tests**:
   - Test the full authentication flow from frontend to backend
   - Verify that authorized requests with valid JWTs are processed correctly
   - Test token renewal and handling of token expiration

3. **Security Tests**:
   - Attempt to access protected resources without authentication
   - Attempt to access protected resources with invalid or expired tokens
   - Attempt to access another user's resources with a valid token
   - Verify that token tampering is detected and rejected

## Conclusion

This authentication integration plan provides a secure, robust way to connect the GoodTalk frontend with the Analyzer Service using industry-standard JWT authentication. While it requires slightly more implementation effort than a simple header-based approach, it provides significantly stronger security guarantees that don't rely on perfect network configuration.

This approach aligns with our iterative programming principle by implementing a solution that is both practical for immediate use and avoids creating security technical debt that would need to be addressed later. The JWT-based approach gives us a solid foundation that will scale well as our application grows and evolves.
