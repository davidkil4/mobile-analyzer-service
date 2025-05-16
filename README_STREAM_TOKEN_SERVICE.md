# Stream Chat Token Service for GoodTalk Mobile

This service provides secure token generation for Stream Chat in the GoodTalk mobile application. It follows an iterative development approach, starting with a simple implementation that can be refined and enhanced as needed.

## Overview

The Stream Chat Token Service is a FastAPI application that generates Stream Chat tokens for authenticated users. It is designed to be used by the GoodTalk mobile app to securely connect to Stream Chat without exposing the Stream API Secret in the client-side code.

## Getting Started

### Prerequisites

- Python 3.8+
- Stream Chat account with API Key and Secret

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Configure the environment variables:

Edit the `.env` file and add your Stream API credentials:

```
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

### Running the Service

Run the service using Uvicorn:

```bash
python stream_token_service.py
```

Or directly with Uvicorn:

```bash
uvicorn stream_token_service:app --reload
```

The service will be available at `http://localhost:8000`.

## API Endpoints

### Generate Stream Token

```
POST /stream/token
```

Generates a Stream Chat token for the authenticated user.

**Authentication:**
- Bearer token (JWT) required in the Authorization header

**Response:**
```json
{
  "token": "stream_chat_token_string",
  "user_id": "authenticated_user_id"
}
```

## Integration with GoodTalk Mobile App

In the GoodTalk mobile app, update the Stream Chat provider to fetch tokens from this service instead of generating them client-side:

```typescript
// In src/providers/stream-chat-provider.tsx

// Replace the current development token generation:
// const token = chatClient.devToken(userId);

// With this secure implementation:
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
    throw error;
  }
};

// Then in your useEffect:
const token = await generateStreamToken(userId);
```

## Security Considerations

1. **JWT Verification**: The current implementation uses a mock JWT verification. In production, implement proper JWT verification using Clerk's public keys.
2. **CORS Configuration**: Update the CORS settings to only allow requests from your mobile app's domain.
3. **Error Handling**: Implement more detailed error handling and logging.
4. **Rate Limiting**: Consider adding rate limiting to prevent abuse.

## Development Roadmap

Following the iterative development approach:

1. **Basic Implementation** (Current)
   - Simple token generation endpoint
   - Mock JWT verification

2. **Enhanced Security**
   - Implement proper JWT verification with Clerk
   - Add proper error handling and logging
   - Configure CORS for production

3. **Performance Optimization**
   - Add token caching
   - Implement rate limiting

4. **Production Readiness**
   - Add monitoring and alerting
   - Comprehensive testing
   - Documentation updates

## Resources

- [Stream Chat Server SDK for Python](https://github.com/GetStream/stream-chat-python)
- [Stream Chat Authentication Documentation](https://getstream.io/chat/docs/python/tokens_and_authentication/)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [Clerk Backend API Documentation](https://clerk.dev/docs/backend-api)
