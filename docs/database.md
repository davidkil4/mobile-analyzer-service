# Database Implementation Guide

## Overview

This document describes the database implementation for storing user-saved teaching modules in the Analyzer Service. The database is designed with a "migration-friendly" approach - starting with SQLite for development and easy transition to PostgreSQL for production.

## Design Principles

1. **Start Simple, Scale Later**: SQLite for development, PostgreSQL for production
2. **Database Abstraction**: Use SQLAlchemy ORM to abstract database specifics
3. **Migration-Friendly Schema**: Avoid SQLite-specific features
4. **Privacy-First**: Only store teaching modules explicitly saved by users (no chat history)
5. **Clean Separation**: Modular code structure with clear separation of concerns

## Directory Structure

```
database/
├── __init__.py       # Package exports
├── api.py           # FastAPI endpoints
├── crud.py          # Database operations (Create, Read, Update, Delete)
├── db.py            # Database connection and session management
└── models.py        # SQLAlchemy ORM models
```

## Database Models

### User Model

Represents a user of the system who can save teaching modules.

```python
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)  # External user ID
    username = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    saved_modules = relationship("SavedModule", back_populates="user", cascade="all, delete-orphan")
```

### SavedModule Model

Represents a teaching module saved by a user.

```python
class SavedModule(Base):
    __tablename__ = 'saved_modules'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    module_id = Column(String(100), nullable=False)  # Original module ID
    module_type = Column(String(50), nullable=False)  # e.g., "grammar", "vocabulary"
    title = Column(String(200), nullable=True)
    focus_type = Column(String(50), nullable=True)  # e.g., "GRAMMAR", "VOCABULARY"
    original_utterance = Column(Text, nullable=True)
    corrected_utterance = Column(Text, nullable=True)
    
    # Store the full module content as JSON
    content = Column(JSON, nullable=False)
    
    # Metadata
    is_favorite = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)  # User's personal notes
    saved_at = Column(DateTime, default=datetime.utcnow)
    last_practiced_at = Column(DateTime, nullable=True)
    practice_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="saved_modules")
```

## Database Connection

The database connection is abstracted through `db.py`, which provides:

1. **Environment-Based Configuration**: Uses `DB_URL` environment variable or defaults to SQLite
2. **Session Management**: Context managers for database sessions
3. **Async Support**: For PostgreSQL with asyncpg
4. **Connection Pooling**: For PostgreSQL scalability

```python
# Default to SQLite for development
DEFAULT_DB_URL = "sqlite:///database/teaching_modules.db"

# Environment variable to control database connection
DB_URL = os.environ.get("DB_URL", DEFAULT_DB_URL)

# Create engine based on DB_URL
if DB_URL.startswith("sqlite"):
    # SQLite-specific settings
    engine = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    # PostgreSQL or other database settings
    engine = create_engine(
        DB_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
    )
```

## CRUD Operations

Database operations are implemented in `crud.py`, providing:

1. **User Management**: Create, get, and get-or-create operations
2. **Module Management**: Save, retrieve, update, and delete operations
3. **Error Handling**: Proper exception handling and transaction management

Example: Saving a module

```python
def save_module(
    db: Session, 
    user_id: str, 
    module_id: str,
    module_type: str,
    content: Dict[str, Any],
    title: Optional[str] = None,
    # ... other parameters
) -> SavedModule:
    """Save a teaching module for a user."""
    # Get or create user
    user = get_or_create_user(db, user_id)
    
    # Check if module is already saved
    existing = get_saved_module_by_user_and_module(db, user_id, module_id)
    if existing:
        # Update existing saved module
        existing.content = content
        # ... update other fields
        db.commit()
        db.refresh(existing)
        return existing
    
    # Create new saved module
    saved_module = SavedModule(
        user_id=user.id,
        module_id=module_id,
        # ... other fields
    )
    
    db.add(saved_module)
    db.commit()
    db.refresh(saved_module)
    return saved_module
```

## API Endpoints

The API endpoints are defined in `api.py` using FastAPI:

1. **POST `/user_modules/save`**: Save a teaching module
2. **GET `/user_modules/user/{user_id}`**: Get all modules saved by a user
3. **PUT `/user_modules/{module_id}`**: Update a saved module
4. **DELETE `/user_modules/{module_id}`**: Delete a saved module

Example: Save endpoint

```python
@router.post("/save", response_model=SavedModuleResponse)
def save_teaching_module(request: SaveModuleRequest = Body(...)):
    """
    Save a teaching module for a user.
    
    If the user has already saved this module, it will be updated.
    """
    try:
        with get_db_session() as db:
            saved = save_module(
                db=db,
                user_id=request.user_id,
                # ... other fields
            )
            return SavedModuleResponse.from_orm(saved)
    except Exception as e:
        logger.error(f"Error saving module: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving module: {str(e)}")
```

## Integration with FastAPI Server

The database is integrated with the main FastAPI server in `api_server.py`:

```python
# Import database modules
from database.db import init_db
from database.api import router as db_router

# Initialize the database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {e}")

# Include the database router
app.include_router(db_router)
```

## Testing

A test script `test_database.py` is provided to verify the database functionality:

```python
def test_database():
    """Run a simple test of the database functionality."""
    init_db()
    
    with get_db_session() as db:
        # Create user
        user = get_or_create_user(db, user_id)
        
        # Save module
        saved_module = save_module(db, user_id, module_id, ...)
        
        # Retrieve modules
        user_modules = get_user_saved_modules(db, user_id)
        
        # Update module
        updated_module = update_saved_module(db, saved_module.id, ...)
        
        # Delete module
        success = delete_saved_module(db, saved_module.id)
```

## Migration to PostgreSQL

When ready to migrate to PostgreSQL:

1. **Update Environment Variable**: Set `DB_URL` to your PostgreSQL connection string
   ```
   DB_URL=postgresql://username:password@localhost:5432/analyzer_service
   ```

2. **Install PostgreSQL Driver**:
   ```
   pip install psycopg2-binary
   ```

3. **For Async Support**:
   ```
   pip install asyncpg
   ```

4. **Run Database Initialization**:
   ```python
   from database.db import init_db
   init_db()
   ```

## Performance Considerations

- **SQLite Limitations**: 
  - Single-writer, multiple-reader
  - Good for ~1,000 concurrent users
  - Up to ~10-20GB data

- **When to Migrate to PostgreSQL**:
  - More than 1,000 concurrent users
  - Data size exceeds 10-20GB
  - Need for better concurrent write performance
  - Production deployment

## Security Best Practices

1. **No Raw Password Storage**: User authentication should be handled separately
2. **Environment Variables**: Database credentials should be in environment variables
3. **Input Validation**: All API inputs are validated with Pydantic models
4. **Error Handling**: Proper error handling to prevent information leakage
5. **Parameterized Queries**: SQLAlchemy ORM prevents SQL injection

## Frontend Integration Guide

### Implementing "Save" Buttons

The database is designed to store teaching modules that users explicitly choose to save. Here's how to implement save functionality in the frontend:

#### 1. Teaching Module Save Button

```jsx
// React component example for a teaching module with save button
function TeachingModule({ module, userId }) {
  const [isSaved, setIsSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  async function handleSave() {
    setIsSaving(true);
    try {
      const response = await fetch('/user_modules/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          module_id: module.module_id,
          module_type: module.module_type,
          content: module, // Send the entire module content
          title: `${module.source_utterance_info.focus_type} - ${module.source_utterance_info.original.substring(0, 30)}...`,
          focus_type: module.source_utterance_info.focus_type,
          original_utterance: module.source_utterance_info.original,
          corrected_utterance: module.source_utterance_info.corrected
        }),
      });
      
      if (response.ok) {
        setIsSaved(true);
        showNotification('Module saved successfully!');
      } else {
        throw new Error('Failed to save module');
      }
    } catch (error) {
      console.error('Error saving module:', error);
      showNotification('Failed to save module', 'error');
    } finally {
      setIsSaving(false);
    }
  }
  
  return (
    <div className="teaching-module">
      <h3>{module.source_utterance_info.focus_type}</h3>
      <div className="original-text">{module.source_utterance_info.original}</div>
      <div className="corrected-text">{module.source_utterance_info.corrected}</div>
      
      {/* Module content rendering */}
      <div className="explanations">
        <p>{module.explanations.introduction}</p>
        {/* More content... */}
      </div>
      
      {/* Save button */}
      <button 
        onClick={handleSave} 
        disabled={isSaving || isSaved}
        className={isSaved ? 'saved-button' : 'save-button'}
      >
        {isSaving ? 'Saving...' : isSaved ? 'Saved ✓' : 'Save for Later'}
      </button>
    </div>
  );
}
```

#### 2. Save Button for Other Content Types

The same database can be used to save other types of content. Just adapt the payload structure:

```jsx
function GrammarRule({ rule, userId }) {
  async function handleSave() {
    try {
      await fetch('/user_modules/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          module_id: `grammar_rule_${rule.id}`,
          module_type: 'grammar_rule',
          content: rule,
          title: rule.title,
          focus_type: 'GRAMMAR_RULE',
          // Other fields as needed
        }),
      });
      // Handle success
    } catch (error) {
      // Handle error
    }
  }
  
  return (
    <div className="grammar-rule">
      <h3>{rule.title}</h3>
      <p>{rule.description}</p>
      <button onClick={handleSave}>Save Rule</button>
    </div>
  );
}
```

### Displaying Saved Modules

To display a user's saved modules, implement a dedicated page or section:

```jsx
function SavedModulesPage({ userId }) {
  const [savedModules, setSavedModules] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    async function fetchSavedModules() {
      try {
        const response = await fetch(`/user_modules/user/${userId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch saved modules');
        }
        const data = await response.json();
        setSavedModules(data);
      } catch (error) {
        console.error('Error fetching saved modules:', error);
        setError('Failed to load your saved modules. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
    
    fetchSavedModules();
  }, [userId]);
  
  async function handleDelete(moduleId) {
    try {
      const response = await fetch(`/user_modules/${moduleId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Remove from state
        setSavedModules(savedModules.filter(module => module.id !== moduleId));
        showNotification('Module removed successfully');
      } else {
        throw new Error('Failed to delete module');
      }
    } catch (error) {
      console.error('Error deleting module:', error);
      showNotification('Failed to delete module', 'error');
    }
  }
  
  async function handleToggleFavorite(module) {
    try {
      const response = await fetch(`/user_modules/${module.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          is_favorite: !module.is_favorite,
        }),
      });
      
      if (response.ok) {
        // Update in state
        setSavedModules(savedModules.map(m => 
          m.id === module.id ? {...m, is_favorite: !m.is_favorite} : m
        ));
      } else {
        throw new Error('Failed to update module');
      }
    } catch (error) {
      console.error('Error updating module:', error);
      showNotification('Failed to update module', 'error');
    }
  }
  
  if (isLoading) return <div>Loading your saved modules...</div>;
  if (error) return <div className="error-message">{error}</div>;
  if (savedModules.length === 0) return <div>You haven't saved any modules yet.</div>;
  
  return (
    <div className="saved-modules-page">
      <h2>Your Saved Modules</h2>
      
      <div className="module-filters">
        <button onClick={() => /* Filter logic */}>All</button>
        <button onClick={() => /* Filter logic */}>Favorites</button>
        <button onClick={() => /* Filter logic */}>Grammar</button>
        <button onClick={() => /* Filter logic */}>Vocabulary</button>
      </div>
      
      <div className="modules-grid">
        {savedModules.map(module => (
          <div key={module.id} className="saved-module-card">
            <h3>{module.title}</h3>
            <p>{module.original_utterance}</p>
            
            <div className="module-actions">
              <button 
                onClick={() => handleToggleFavorite(module)}
                className={module.is_favorite ? 'favorite active' : 'favorite'}
              >
                {module.is_favorite ? '★' : '☆'}
              </button>
              
              <button onClick={() => /* View details */}>View</button>
              <button onClick={() => handleDelete(module.id)}>Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### User Authentication Integration

The database includes a User model that can be integrated with your authentication system:

#### 1. After User Registration/Login

When a user registers or logs in through your authentication system (e.g., Firebase, Auth0, or custom auth), create or retrieve the user in the database:

```jsx
async function handleLoginSuccess(authUser) {
  // authUser comes from your auth provider (e.g., Firebase)
  try {
    // Call your backend to create/get the user in the database
    const response = await fetch('/api/sync_user', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: authUser.uid, // External auth ID
        username: authUser.displayName,
        email: authUser.email,
      }),
    });
    
    if (!response.ok) {
      console.error('Failed to sync user with database');
    }
    
    // Continue with login flow
    setUser(authUser);
    navigate('/dashboard');
  } catch (error) {
    console.error('Error syncing user:', error);
    // Handle error (but still allow login)
    setUser(authUser);
    navigate('/dashboard');
  }
}
```

#### 2. Backend User Sync Endpoint

Implement a backend endpoint to sync the authenticated user with the database:

```python
# In api_server.py or a dedicated auth.py file

from fastapi import APIRouter, Depends, HTTPException, Body
from database.db import get_db_session
from database.crud import get_or_create_user

auth_router = APIRouter(prefix="/api", tags=["auth"])

class UserSyncRequest(BaseModel):
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None

@auth_router.post("/sync_user")
def sync_user(request: UserSyncRequest = Body(...)):
    """Sync a user from the authentication system with the database."""
    try:
        with get_db_session() as db:
            user = get_or_create_user(
                db, 
                request.user_id, 
                request.username, 
                request.email
            )
            return {"status": "success", "user_id": user.user_id}
    except Exception as e:
        logger.error(f"Error syncing user: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing user: {str(e)}")

# Add to api_server.py
app.include_router(auth_router)
```

### Practical Implementation Tips

1. **User ID Consistency**: Always use the same user ID from your authentication system when interacting with the database.

2. **Error Handling**: Implement robust error handling in the frontend for database operations.

3. **Loading States**: Show appropriate loading indicators during database operations.

4. **Optimistic Updates**: Update the UI optimistically before the server confirms the operation, then revert if it fails.

5. **Caching**: Consider caching saved modules in local storage for faster loading and offline access.

6. **Pagination**: Implement pagination for users with many saved modules.

7. **Search and Filter**: Add search and filter functionality for the saved modules page.

8. **Sync Indicator**: Show a sync status indicator for save operations.

```jsx
// Example of a reusable save button component
function SaveButton({ item, itemType, userId, onSaveComplete }) {
  const [saveStatus, setSaveStatus] = useState('idle'); // 'idle', 'saving', 'saved', 'error'
  
  async function handleSave() {
    setSaveStatus('saving');
    try {
      // Prepare the payload based on item type
      const payload = {
        user_id: userId,
        module_id: item.id || `${itemType}_${Date.now()}`,
        module_type: itemType,
        content: item,
        // Other fields based on item type
      };
      
      // Add type-specific fields
      if (itemType === 'teaching_module') {
        payload.title = `${item.source_utterance_info.focus_type} - ${item.source_utterance_info.original.substring(0, 30)}...`;
        payload.focus_type = item.source_utterance_info.focus_type;
        payload.original_utterance = item.source_utterance_info.original;
        payload.corrected_utterance = item.source_utterance_info.corrected;
      } else if (itemType === 'grammar_rule') {
        payload.title = item.title;
        payload.focus_type = 'GRAMMAR_RULE';
      }
      
      const response = await fetch('/user_modules/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (response.ok) {
        setSaveStatus('saved');
        if (onSaveComplete) onSaveComplete(await response.json());
      } else {
        throw new Error('Failed to save item');
      }
    } catch (error) {
      console.error('Error saving item:', error);
      setSaveStatus('error');
    }
  }
  
  return (
    <button 
      onClick={handleSave} 
      disabled={saveStatus === 'saving' || saveStatus === 'saved'}
      className={`save-button ${saveStatus}`}
    >
      {saveStatus === 'idle' && 'Save'}
      {saveStatus === 'saving' && 'Saving...'}
      {saveStatus === 'saved' && 'Saved ✓'}
      {saveStatus === 'error' && 'Retry'}
    </button>
  );
}
```
