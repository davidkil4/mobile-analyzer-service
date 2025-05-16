# Analyzer Service API Endpoints - Backend Implementation

This document provides technical details about the API endpoints implemented in the Analyzer Service backend. It includes schema definitions, validation rules, database interactions, and implementation notes.

## Integration with Frontend

The GoodTalk frontend communicates with this backend through proxy routes. The frontend proxy routes map to these backend endpoints as follows:

| Frontend Endpoint | Method | Backend Endpoint |
|-------------------|--------|------------------|
| `/api/analyzer` | POST | `/user_modules/save` |
| `/api/analyzer` | GET | `/user_modules/user/{user_id}` |
| `/api/analyzer/[id]` | PUT | `/user_modules/{module_id}` |
| `/api/analyzer/[id]` | DELETE | `/user_modules/{module_id}` |

## Authentication

All endpoints are protected using JWT authentication. The JWT token is verified using the following process:

1. Extract the JWT token from the `Authorization` header
2. Verify the token's signature using the Clerk public key
3. Extract the user ID from the verified token
4. Use the user ID for database operations

## Database Models

### User Model

```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, index=True)  # External user ID (from auth provider)
    username = Column(String, nullable=True)
    
    # Relationship to saved modules
    saved_modules = relationship("SavedModule", back_populates="user", cascade="all, delete-orphan")
```

### SavedModule Model

```python
class SavedModule(Base):
    __tablename__ = "saved_modules"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    module_id = Column(String, index=True)  # Original module ID
    module_type = Column(String)
    title = Column(String, nullable=True)
    focus_type = Column(String, nullable=True)
    original_utterance = Column(String, nullable=True)
    corrected_utterance = Column(String, nullable=True)
    content = Column(JSON)  # Full module content as JSON
    is_favorite = Column(Boolean, default=False)
    notes = Column(String, nullable=True)
    saved_at = Column(DateTime, default=func.now())
    last_practiced_at = Column(DateTime, nullable=True)
    practice_count = Column(Integer, default=0)
    
    # Relationship to user
    user = relationship("User", back_populates="saved_modules")
```

## Pydantic Schemas

### SavedModuleRequest

```python
class SaveModuleRequest(BaseModel):
    user_id: str = Field(..., description="User ID from the authentication system")
    module_id: str = Field(..., description="Original module ID from the teaching module")
    module_type: str = Field(..., description="Type of module (grammar, vocabulary, pattern)")
    content: Dict[str, Any] = Field(..., description="Full module content as JSON")
    title: Optional[str] = Field(None, description="Module title")
    focus_type: Optional[str] = Field(None, description="Focus type (GRAMMAR, VOCABULARY, PATTERN)")
    original_utterance: Optional[str] = Field(None, description="Original utterance text")
    corrected_utterance: Optional[str] = Field(None, description="Corrected utterance text")
    is_favorite: bool = Field(False, description="Whether this module is marked as favorite")
    notes: Optional[str] = Field(None, description="User's notes about this module")
```

### SavedModuleResponse

```python
class SavedModuleResponse(BaseModel):
    id: int
    user_id: str  # External user ID (from auth provider)
    module_id: str
    module_type: str
    title: Optional[str] = None
    focus_type: Optional[str] = None
    original_utterance: Optional[str] = None
    corrected_utterance: Optional[str] = None
    content: Dict[str, Any]
    is_favorite: bool
    notes: Optional[str] = None
    saved_at: datetime
    last_practiced_at: Optional[datetime] = None
    practice_count: int
    
    # Ensure user_id is always a string, even if it comes from the database as an integer
    @validator('user_id', pre=True)
    def ensure_string_user_id(cls, v):
        return str(v) if v is not None else None
    
    class Config:
        orm_mode = True  # For Pydantic v1 compatibility
        from_attributes = True  # For Pydantic v2 compatibility
```

### UpdateModuleRequest

```python
class UpdateModuleRequest(BaseModel):
    is_favorite: Optional[bool] = Field(None, description="Whether this module is marked as favorite")
    notes: Optional[str] = Field(None, description="User's notes about this module")
    practice_count: Optional[int] = Field(None, description="Number of times practiced")
```

## API Endpoints

### 1. Save Teaching Module

**Endpoint:** `/user_modules/save`  
**Method:** `POST`  
**Implementation:**

```python
@router.post("/save", response_model=SavedModuleResponse)
def save_teaching_module(
    request: SaveModuleRequest = Body(...),
):
    """
    Save a teaching module for a specific user.
    
    Args:
        module: The module data to save
        user_id: The authenticated user's ID (from JWT)
        
    Returns:
        The saved module with database ID and metadata
    """
    with get_db_session() as db:
        # Get or create the user
        user = get_or_create_user(db, user_id)
        
        # Save the module
        saved_module = save_module(
            db, 
            user_id=user.id,
            module_id=module.module_id,
            module_type=module.module_type,
            title=module.title,
            focus_type=module.focus_type,
            original_utterance=module.original_utterance,
            corrected_utterance=module.corrected_utterance,
            content=module.content
        )
        
        # Convert internal user ID to external user ID for the response
        response = SavedModuleResponse.from_orm(saved_module)
        response.user_id = user_id
        
        return response
```

### 2. Get User's Saved Modules

**Endpoint:** `/user_modules/user/{user_id}`  
**Method:** `GET`  
**Implementation:**

```python
@router.get("/user/{user_id}", response_model=List[SavedModuleResponse])
def get_user_modules(
    user_id: str = Path(..., description="User ID from the authentication system"),
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(100, description="Maximum number of items to return"),
):
    """
    Get all teaching modules saved by a specific user.
    
    Args:
        user_id: The authenticated user's ID (from JWT)
        
    Returns:
        List of saved modules with their metadata
    """
    with get_db_session() as db:
        # Get or create the user
        user = get_or_create_user(db, user_id)
        
        # Get the user's saved modules
        modules = get_user_saved_modules(db, user.id)
        
        # Convert internal user IDs to external user ID for the response
        responses = []
        for module in modules:
            response = SavedModuleResponse.from_orm(module)
            response.user_id = user_id
            responses.append(response)
        
        return responses
```

### 3. Update a Saved Module

**Endpoint:** `/user_modules/{module_id}`  
**Method:** `PUT`  
**Implementation:**

```python
@router.put("/{module_id}", response_model=SavedModuleResponse)
def update_module(
    module_id: int = Path(..., description="ID of the saved module to update"),
    request: UpdateModuleRequest = Body(...),
):
    """
    Update properties of a saved teaching module.
    
    Args:
        module_id: The database ID of the module to update
        update_data: The data to update
        user_id: The authenticated user's ID (from JWT)
        
    Returns:
        The updated module with its metadata
    """
    with get_db_session() as db:
        # Get or create the user
        user = get_or_create_user(db, user_id)
        
        # Update the module
        updated_module = update_saved_module(
            db,
            module_id=module_id,
            user_id=user.id,
            is_favorite=update_data.is_favorite,
            notes=update_data.notes,
            practice_count=update_data.practice_count
        )
        
        if not updated_module:
            raise HTTPException(
                status_code=404,
                detail=f"Module with ID {module_id} not found for this user"
            )
        
        # Convert internal user ID to external user ID for the response
        response = SavedModuleResponse.from_orm(updated_module)
        response.user_id = user_id
        
        return response
```

### 4. Delete a Saved Module

**Endpoint:** `/user_modules/{module_id}`  
**Method:** `DELETE`  
**Implementation:**

```python
@router.delete("/{module_id}")
def delete_module(
    module_id: int = Path(..., description="ID of the saved module to delete"),
):
    """
    Delete a saved teaching module.
    
    Args:
        module_id: The database ID of the module to delete
        user_id: The authenticated user's ID (from JWT)
        
    Returns:
        Success message
    """
    with get_db_session() as db:
        # Get or create the user
        user = get_or_create_user(db, user_id)
        
        # Delete the module
        success = delete_saved_module(db, module_id=module_id, user_id=user.id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Module with ID {module_id} not found for this user"
            )
        
        return {"success": True, "message": f"Module with ID {module_id} deleted successfully"}
```

## Database Operations

### get_or_create_user

```python
def get_or_create_user(db: Session, user_id: str, username: Optional[str] = None) -> User:
    """
    Get a user by their external user ID, or create a new user if they don't exist.
    
    Args:
        db: Database session
        user_id: External user ID (from auth provider)
        username: Optional username
        
    Returns:
        User object
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    
    if not user:
        # Create a new user
        user = User(user_id=user_id, username=username or f"User {user_id}")
        db.add(user)
        db.commit()
        db.refresh(user)
    
    return user
```

### save_module

```python
def save_module(
    db: Session,
    user_id: int,
    module_id: str,
    module_type: str,
    content: Dict[str, Any],
    title: Optional[str] = None,
    focus_type: Optional[str] = None,
    original_utterance: Optional[str] = None,
    corrected_utterance: Optional[str] = None
) -> SavedModule:
    """
    Save a teaching module for a user.
    
    Args:
        db: Database session
        user_id: Internal user ID
        module_id: Original module ID
        module_type: Type of module
        content: Full module content as JSON
        title: Optional title
        focus_type: Optional focus type
        original_utterance: Optional original utterance
        corrected_utterance: Optional corrected utterance
        
    Returns:
        Saved module object
    """
    # Create a new saved module
    saved_module = SavedModule(
        user_id=user_id,
        module_id=module_id,
        module_type=module_type,
        title=title,
        focus_type=focus_type,
        original_utterance=original_utterance,
        corrected_utterance=corrected_utterance,
        content=content
    )
    
    db.add(saved_module)
    db.commit()
    db.refresh(saved_module)
    
    return saved_module
```

### get_user_saved_modules

```python
def get_user_saved_modules(db: Session, user_id: int) -> List[SavedModule]:
    """
    Get all teaching modules saved by a user.
    
    Args:
        db: Database session
        user_id: Internal user ID
        
    Returns:
        List of saved module objects
    """
    return db.query(SavedModule).filter(SavedModule.user_id == user_id).all()
```

### update_saved_module

```python
def update_saved_module(
    db: Session,
    module_id: int,
    user_id: int,
    is_favorite: Optional[bool] = None,
    notes: Optional[str] = None,
    practice_count: Optional[int] = None
) -> Optional[SavedModule]:
    """
    Update properties of a saved teaching module.
    
    Args:
        db: Database session
        module_id: Database ID of the module
        user_id: Internal user ID
        is_favorite: Whether the module is marked as favorite
        notes: User's notes about the module
        practice_count: Number of times practiced
        
    Returns:
        Updated module object, or None if not found
    """
    # Get the module
    module = db.query(SavedModule).filter(
        SavedModule.id == module_id,
        SavedModule.user_id == user_id
    ).first()
    
    if not module:
        return None
    
    # Update the module properties
    if is_favorite is not None:
        module.is_favorite = is_favorite
    
    if notes is not None:
        module.notes = notes
    
    if practice_count is not None:
        module.practice_count = practice_count
        module.last_practiced_at = func.now()
    
    db.commit()
    db.refresh(module)
    
    return module
```

### delete_saved_module

```python
def delete_saved_module(db: Session, module_id: int, user_id: int) -> bool:
    """
    Delete a saved teaching module.
    
    Args:
        db: Database session
        module_id: Database ID of the module
        user_id: Internal user ID
        
    Returns:
        True if the module was deleted, False if not found
    """
    # Get the module
    module = db.query(SavedModule).filter(
        SavedModule.id == module_id,
        SavedModule.user_id == user_id
    ).first()
    
    if not module:
        return False
    
    # Delete the module
    db.delete(module)
    db.commit()
    
    return True
```

## Error Handling

All endpoints use FastAPI's built-in exception handling. Common exceptions include:

1. **HTTPException(status_code=401)**: Returned when authentication fails.
2. **HTTPException(status_code=403)**: Returned when the user doesn't have permission.
3. **HTTPException(status_code=404)**: Returned when a resource doesn't exist.
4. **HTTPException(status_code=400)**: Returned when validation fails.
5. **HTTPException(status_code=500)**: Returned for unexpected server errors.

## Performance Considerations

1. **Database Indexing**: The `user_id` and `module_id` columns are indexed for faster queries.
2. **Connection Pooling**: Database connections are pooled for better performance.
3. **Session Management**: Database sessions are properly closed using context managers.
4. **Lazy Loading**: Relationships are lazy-loaded to avoid unnecessary database queries.

## Security Considerations

1. **JWT Verification**: All tokens are cryptographically verified.
2. **User Isolation**: Users can only access their own data.
3. **Input Validation**: All input is validated using Pydantic models.
4. **SQL Injection Prevention**: SQLAlchemy ORM prevents SQL injection.
5. **Error Messages**: Error messages don't reveal sensitive information.

## Pydantic Version Compatibility

All Pydantic models in this codebase support both Pydantic v1 and v2 through dual configuration:

```python
class Config:
    orm_mode = True  # For Pydantic v1 compatibility
    from_attributes = True  # For Pydantic v2 compatibility
```

This ensures backward compatibility while allowing for future upgrades. When using these models, be aware of the following:

1. For ORM model conversion, both `from_orm()` (v1) and the constructor (v2) will work
2. The `user_id` field is automatically converted to a string, even if it's an integer in the database
3. All fields are validated according to their type annotations
