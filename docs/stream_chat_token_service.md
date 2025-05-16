# Stream Chat Token Service Implementation

This document outlines the implementation of the Stream Chat Token Service for the GoodTalk mobile application. The service provides secure token generation for Stream Chat, following best practices for authentication and security.

## Table of Contents

1. [Overview](#overview)
2. [Implementation Approach](#implementation-approach)
3. [Components](#components)
4. [Authentication Flow](#authentication-flow)
5. [Development and Testing](#development-and-testing)
6. [Frontend Integration](#frontend-integration)
7. [Production Deployment](#production-deployment)
8. [Security Considerations](#security-considerations)

## Overview

The Stream Chat Token Service is a FastAPI application that generates secure tokens for Stream Chat in the GoodTalk mobile application. It follows an iterative development approach, starting with a simple implementation and gradually adding more features and security.

### Problem Statement

In the initial implementation of the GoodTalk mobile app, Stream Chat tokens were generated client-side using the development token method:

```typescript
// DEVELOPMENT ONLY - NOT SECURE FOR PRODUCTION
const token = chatClient.devToken(userId);
```

This approach is not secure for production as it:
- Only works when "Disable Auth Checks" is enabled in the Stream Dashboard
- Does not properly authenticate users
- Exposes the application to potential security vulnerabilities

### Solution

The Stream Chat Token Service solves this problem by:
- Moving token generation to the server-side
- Implementing proper JWT verification with Clerk
- Keeping the Stream API Secret secure on the server
- Providing a robust authentication flow

## Implementation Approach

Following the iterative development approach, the implementation was done in stages:

### Stage 1: Basic Implementation

- Created a simple FastAPI application for token generation
- Implemented mock JWT verification for development
- Added basic error handling and logging
- Set up environment configuration

### Stage 2: Enhanced Security

- Implemented proper JWT verification with Clerk
- Added JWKs caching for performance
- Enhanced error handling for different JWT verification failures
- Maintained backward compatibility with development mode

### Stage 3: Testing and Refinement

- Created test scripts for both mock and real JWT tokens
- Added detailed logging for debugging
- Implemented fallback mechanisms for development

## Components

### 1. Stream Token Service (`stream_token_service.py`)

The main service that:
- Verifies JWT tokens from Clerk
- Generates Stream Chat tokens
- Handles errors and logging

Key features:
- JWT verification with Clerk
- JWKs caching for performance
- Development mode for easier testing
- Detailed error handling

### 2. Environment Configuration (`.env`)

Stores configuration variables:
- Stream API credentials
- Clerk API credentials
- Development mode flag

### 3. Test Scripts

#### Basic Test Script (`test_stream_token.py`)

Simple script to test token generation with a mock JWT token.

#### Enhanced Test Script (`test_stream_token_enhanced.py`)

More advanced script that supports:
- Testing with mock JWT tokens
- Testing with real JWT tokens
- Command-line arguments for flexibility

#### Clerk Token Generator (`generate_clerk_token.py`)

Utility script to generate test tokens using the Clerk API.

## Authentication Flow

1. **Client Authentication**:
   - User authenticates with Clerk in the mobile app
   - Mobile app obtains a JWT token from Clerk

2. **Token Request**:
   - Mobile app sends a request to the Stream Chat Token Service
   - Request includes the JWT token in the Authorization header

3. **JWT Verification**:
   - Service verifies the JWT token with Clerk
   - Service extracts the user ID from the verified token

4. **Token Generation**:
   - Service generates a Stream Chat token for the authenticated user
   - Token is returned to the mobile app

5. **Stream Chat Initialization**:
   - Mobile app initializes Stream Chat with the generated token
   - User can now use Stream Chat features securely

## Development and Testing

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   ```
   # Stream Chat API credentials
   STREAM_API_KEY=your_stream_api_key
   STREAM_API_SECRET=your_stream_api_secret

   # Clerk API credentials
   CLERK_FRONTEND_API=your_clerk_frontend_api
   CLERK_JWT_ISSUER=your_clerk_jwt_issuer
   CLERK_API_KEY=your_clerk_api_key

   # Set to "true" for development mode
   DEV_MODE=true
   ```

3. Run the service:
   ```bash
   python stream_token_service.py
   ```

### Testing

#### With Mock JWT Token

```bash
python test_stream_token.py
```

#### With Real JWT Token

```bash
python test_stream_token_enhanced.py --token "your_real_jwt_token"
```

or

```bash
python test_stream_token_enhanced.py --token-file jwt_token.txt
```

## Frontend Integration

To integrate with the mobile app, update the Stream Chat provider:

```typescript
// In src/providers/stream-chat-provider.tsx

// Replace the current development token generation:
// const token = chatClient.devToken(userId);

// With this secure implementation:
const token = await generateStreamToken(userId);

// Add this function to your provider:
const generateStreamToken = async (userId: string): Promise<string> => {
  try {
    // Get the Clerk session token
    const sessionToken = await getToken(); // From Clerk's useAuth() hook
    
    if (!sessionToken) {
      throw new Error('No authentication token available');
    }
    
    // Call your backend API to get a Stream token
    const response = await fetch('YOUR_BACKEND_URL/stream/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`,
      },
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get Stream token: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.token;
  } catch (error) {
    console.error('Error generating Stream token:', error);
    // For development, fall back to client-side token generation if server fails
    if (__DEV__) {
      console.warn('Falling back to client-side token generation (DEV ONLY)');
      return chatClient.devToken(userId);
    }
    throw error;
  }
};
```

## Production Deployment

For production deployment:

1. Deploy the service to a secure environment
2. Set up HTTPS for all API requests
3. Configure proper CORS settings
4. Set `DEV_MODE=false` in the environment
5. Ensure all secrets are properly managed
6. Turn off "Disable Auth Checks" in the Stream Dashboard

## Security Considerations

1. **JWT Verification**: Always verify JWT tokens properly before generating Stream Chat tokens
2. **API Secret Protection**: Never expose the Stream API Secret in client-side code
3. **HTTPS**: Use HTTPS for all API requests to prevent token interception
4. **CORS**: Configure proper CORS settings to prevent unauthorized access
5. **Error Handling**: Implement proper error handling to prevent information leakage
6. **Rate Limiting**: Consider adding rate limiting to prevent abuse
7. **Monitoring**: Implement monitoring and alerting for security events

## Conclusion

The Stream Chat Token Service provides a secure and robust solution for generating Stream Chat tokens in the GoodTalk mobile application. By following best practices for authentication and security, it ensures that user data is protected and that the application is ready for production use.

By implementing this service, we've addressed the security concerns of the initial client-side token generation approach and provided a foundation for secure Stream Chat integration in the mobile app.
