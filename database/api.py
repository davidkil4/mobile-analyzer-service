"""
API endpoints for user-saved teaching modules.

This module provides FastAPI endpoints for saving, retrieving, updating, and deleting
user-saved teaching modules.
"""
from fastapi import APIRouter, Depends, HTTPException, Body, Path, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import logging
import json
from datetime import datetime

from database.db import get_db_session
from database.crud import (
    get_user_saved_modules,
    save_module,
    get_saved_module,
    update_saved_module,
    delete_saved_module
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/user_modules",
    tags=["user_modules"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request/response validation
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

class SavedModuleResponse(BaseModel):
    id: int
    user_id: str
    module_id: str
    module_type: str
    title: Optional[str]
    focus_type: Optional[str]
    original_utterance: Optional[str]
    corrected_utterance: Optional[str]
    content: Dict[str, Any]
    is_favorite: bool
    notes: Optional[str]
    saved_at: datetime
    last_practiced_at: Optional[datetime]
    practice_count: int

    # Ensure user_id is always a string, even if it comes from the database as an integer
    @validator('user_id', pre=True)
    def ensure_string_user_id(cls, v):
        return str(v) if v is not None else None

    class Config:
        orm_mode = True  # For Pydantic v1 compatibility
        from_attributes = True  # For Pydantic v2 compatibility

class UpdateModuleRequest(BaseModel):
    is_favorite: Optional[bool] = Field(None, description="Whether this module is marked as favorite")
    notes: Optional[str] = Field(None, description="User's notes about this module")
    practice_count: Optional[int] = Field(None, description="Number of times practiced")

# API Endpoints
@router.post("/save", response_model=SavedModuleResponse)
def save_teaching_module(
    request: SaveModuleRequest = Body(...),
):
    """
    Save a teaching module for a user.
    
    If the user has already saved this module, it will be updated.
    
    This endpoint should be called when a user clicks a "Save" button on a teaching module.
    """
    try:
        with get_db_session() as db:
            saved = save_module(
                db=db,
                user_id=request.user_id,
                module_id=request.module_id,
                module_type=request.module_type,
                content=request.content,
                title=request.title,
                focus_type=request.focus_type,
                original_utterance=request.original_utterance,
                corrected_utterance=request.corrected_utterance,
                is_favorite=request.is_favorite,
                notes=request.notes
            )
            
            # Convert SQLAlchemy model to Pydantic model
            return SavedModuleResponse.from_orm(saved)
    except Exception as e:
        logger.error(f"Error saving module: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving module: {str(e)}")

@router.get("/user/{user_id}", response_model=List[SavedModuleResponse])
def get_user_modules(
    user_id: str = Path(..., description="User ID from the authentication system"),
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(100, description="Maximum number of items to return"),
):
    """
    Get all teaching modules saved by a user.
    
    This endpoint should be called when loading a user's saved modules page.
    """
    try:
        with get_db_session() as db:
            modules = get_user_saved_modules(db, user_id, skip, limit)
            return [SavedModuleResponse.from_orm(m) for m in modules]
    except Exception as e:
        logger.error(f"Error getting user modules: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting user modules: {str(e)}")

@router.put("/{module_id}", response_model=SavedModuleResponse)
def update_module(
    module_id: int = Path(..., description="ID of the saved module to update"),
    request: UpdateModuleRequest = Body(...),
):
    """
    Update a saved teaching module.
    
    This endpoint allows updating user-specific fields like is_favorite, notes, and practice_count.
    """
    try:
        with get_db_session() as db:
            updated = update_saved_module(
                db=db,
                saved_module_id=module_id,
                is_favorite=request.is_favorite,
                notes=request.notes,
                practice_count=request.practice_count
            )
            
            if not updated:
                raise HTTPException(status_code=404, detail=f"Saved module with ID {module_id} not found")
            
            return SavedModuleResponse.from_orm(updated)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating module: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating module: {str(e)}")

@router.delete("/{module_id}")
def delete_module(
    module_id: int = Path(..., description="ID of the saved module to delete"),
):
    """
    Delete a saved teaching module.
    
    This endpoint should be called when a user clicks a "Delete" button on a saved module.
    """
    try:
        with get_db_session() as db:
            success = delete_saved_module(db, module_id)
            
            if not success:
                raise HTTPException(status_code=404, detail=f"Saved module with ID {module_id} not found")
            
            return {"status": "success", "message": f"Module {module_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting module: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting module: {str(e)}")
