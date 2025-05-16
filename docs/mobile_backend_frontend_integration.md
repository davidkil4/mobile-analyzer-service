# Mobile App to Analyzer Service Integration Plan

## 1. System Overview & Integration Goals

**Current Architecture:**
- **Mobile App (React Native)**: Uses Stream Chat for real-time messaging
- **Stream Token Service**: Handles authentication and channel management for Stream Chat
- **Analyzer Service**: Processes conversation data, performs analysis, and generates teaching modules

**Integration Goals:**
- Send messages from the mobile app's Stream Chat to the analyzer service
- Batch messages by conversation for efficient analysis
- Trigger full conversation analysis when requested by the user
- Display generated teaching modules in the mobile app

## 2. Key Integration Components

### A. Message Flow Integration

1. **Message Interception Point**:
   - Intercept messages in the `CustomMessageInput` component when `sendMessage()` is called
   - This ensures we capture messages at the point of sending to Stream Chat

2. **Conversation ID Management**:
   - Use Stream Chat's channel ID as the conversation ID for the analyzer service
   - This provides a consistent identifier across both systems

3. **Speaker Identification**:
   - Map Stream Chat's user roles to the analyzer's expected speaker roles:
     - Current user → "student"
     - Other users → "interviewer"

### B. Authentication & Security

1. **Token Reuse**:
   - Reuse the existing JWT authentication from Clerk
   - Pass the same authorization header to both Stream Chat and analyzer service

2. **Error Handling**:
   - Implement robust error handling for network failures
   - Ensure message delivery to Stream Chat is prioritized over analyzer service

### C. User Interface Components

1. **Analyze Conversation Button**:
   - Add a button to the channel screen to trigger full conversation analysis
   - Only enable when sufficient messages exist in the conversation

2. **Teaching Module Display**:
   - Create a new screen to display teaching modules
   - Implement a polling mechanism to check for new modules

## 3. Implementation Phases

### Phase 1: Backend Verification & Preparation

**Tasks:**
- [ ] Verify analyzer service API endpoints are ready for mobile integration
- [ ] Test the `/message` endpoint with sample data from mobile format
- [ ] Confirm authentication flow works with mobile app's JWT tokens
- [ ] Ensure CORS is properly configured for mobile app requests

**Success Criteria:**
- Analyzer service accepts and processes messages from test requests
- Authentication mechanism works with mobile app's JWT tokens

### Phase 2: Basic Message Integration

**Tasks:**
- [ ] Create an `AnalyzerService` class in the mobile app
- [ ] Implement the `sendMessage` method to forward messages to the analyzer service
- [ ] Integrate with `CustomMessageInput` to intercept messages
- [ ] Add proper error handling and logging

**Success Criteria:**
- Messages sent in Stream Chat are successfully forwarded to the analyzer service
- The user experience in Stream Chat is not affected by the integration

### Phase 3: Conversation Analysis Trigger

**Tasks:**
- [ ] Add an "Analyze Conversation" button to the channel screen
- [ ] Implement the endpoint call to `/end_conversation/{conversation_id}`
- [ ] Create a loading indicator for analysis in progress
- [ ] Handle the response from the analysis endpoint

**Success Criteria:**
- User can trigger full conversation analysis
- The system correctly processes all messages in the conversation

### Phase 4: Teaching Module Display

**Tasks:**
- [ ] Create a new screen for displaying teaching modules
- [ ] Implement polling to `/teaching_modules/list/{conversation_id}`
- [ ] Fetch and display individual modules using `/teaching_module/{filename}`
- [ ] Add navigation between chat and teaching modules

**Success Criteria:**
- Teaching modules are displayed to the user after analysis
- User can navigate between chat and teaching modules

## 4. Technical Implementation Details

### A. Mobile App Changes

1. **New Files to Create:**
   - `src/services/analyzer-service.ts`: Service for communicating with analyzer backend
   - `src/hooks/useAnalyzer.ts`: Hook for using the analyzer service
   - `src/screens/TeachingModulesScreen.tsx`: Screen for displaying teaching modules
   - `src/components/AnalyzeButton.tsx`: Button component for triggering analysis

2. **Files to Modify:**
   - `src/components/chat/CustomMessageInput.tsx`: Add message interception
   - `app/channel/[id].tsx`: Add analyze button and navigation to teaching modules
   - `src/utils/env.ts`: Add analyzer service URL

### B. Backend Verification

1. **Endpoints to Test:**
   - `POST /message`: Test with sample Stream Chat message format
   - `POST /end_conversation/{conversation_id}`: Test with sample conversation ID
   - `GET /teaching_modules/list/{conversation_id}`: Verify response format
   - `GET /teaching_module/{filename}`: Verify module retrieval

2. **Authentication Flow:**
   - Verify JWT token validation works with mobile app's tokens
   - Test error cases for invalid or expired tokens

## 5. Data Flow Diagrams

### Message Flow:
```
User Types Message → CustomMessageInput → Stream Chat
                                       → Analyzer Service (/message endpoint)
                                          → Batching
                                          → Analysis (every 10 messages)
```

### Conversation Analysis Flow:
```
User Taps "Analyze" → API Call to /end_conversation/{conversation_id}
                    → Analyzer Service processes remaining messages
                    → Clustering Analysis runs
                    → Teaching Module Generation starts
                    → Response with clustering results
                    → Mobile app begins polling for teaching modules
```

## 6. Testing Strategy

1. **Unit Tests:**
   - Test the `AnalyzerService` class methods
   - Verify proper message formatting and error handling

2. **Integration Tests:**
   - Test the full flow from message sending to analyzer service
   - Verify conversation analysis trigger and response handling

3. **End-to-End Tests:**
   - Complete conversation flow from messaging to teaching module display
   - Test with different conversation scenarios (short, long, mixed languages)

## 7. Risks and Mitigations

1. **Network Reliability:**
   - **Risk**: Mobile network instability affecting analyzer service communication
   - **Mitigation**: Implement retry logic and offline queue for messages

2. **Performance Impact:**
   - **Risk**: Sending messages to analyzer service slowing down the chat experience
   - **Mitigation**: Use asynchronous, non-blocking calls to the analyzer service

3. **Authentication Synchronization:**
   - **Risk**: Token expiration or validation issues between systems
   - **Mitigation**: Implement token refresh and consistent error handling

## 8. Next Steps

1. Begin with backend verification to confirm readiness
2. Create the `AnalyzerService` class as the foundation for integration
3. Implement basic message forwarding to test the connection
4. Build the conversation analysis trigger and teaching module display

Following our iterative development approach, we'll start with the simplest working implementation and refine it step by step, ensuring we have a solid foundation before adding more complex features.
