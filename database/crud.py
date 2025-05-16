"""
CRUD operations for database models.

This module provides Create, Read, Update, Delete operations for database models,
ensuring compatibility with both SQLite and PostgreSQL.
"""
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database.models import User, SavedModule

logger = logging.getLogger(__name__)

# User operations
def get_user(db: Session, user_id: str) -> Optional[User]:
    """Get a user by their external user_id."""
    return db.query(User).filter(User.user_id == user_id).first()

def create_user(db: Session, user_id: str, username: Optional[str] = None, email: Optional[str] = None) -> User:
    """Create a new user."""
    db_user = User(user_id=user_id, username=username, email=email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_or_create_user(db: Session, user_id: str, username: Optional[str] = None, email: Optional[str] = None) -> User:
    """Get a user by ID, or create if not exists."""
    user = get_user(db, user_id)
    if not user:
        user = create_user(db, user_id, username, email)
    return user

# SavedModule operations
def get_saved_module(db: Session, saved_module_id: int) -> Optional[SavedModule]:
    """Get a saved module by its ID."""
    return db.query(SavedModule).filter(SavedModule.id == saved_module_id).first()

def get_saved_module_by_user_and_module(db: Session, user_id: str, module_id: str) -> Optional[SavedModule]:
    """Check if a user has already saved a specific module."""
    user = get_user(db, user_id)
    if not user:
        return None
    
    return db.query(SavedModule).filter(
        SavedModule.user_id == user.id,
        SavedModule.module_id == module_id
    ).first()

def get_user_saved_modules(db: Session, user_id: str, skip: int = 0, limit: int = 100) -> List[SavedModule]:
    """Get all saved modules for a user."""
    user = get_user(db, user_id)
    if not user:
        return []
    
    return db.query(SavedModule).filter(
        SavedModule.user_id == user.id
    ).offset(skip).limit(limit).all()

def save_module(
    db: Session, 
    user_id: str, 
    module_id: str,
    module_type: str,
    content: Dict[str, Any],
    title: Optional[str] = None,
    focus_type: Optional[str] = None,
    original_utterance: Optional[str] = None,
    corrected_utterance: Optional[str] = None,
    is_favorite: bool = False,
    notes: Optional[str] = None
) -> SavedModule:
    """Save a teaching module for a user."""
    # Get or create user
    user = get_or_create_user(db, user_id)
    
    # Check if module is already saved
    existing = get_saved_module_by_user_and_module(db, user_id, module_id)
    if existing:
        # Update existing saved module
        existing.content = content
        existing.title = title or existing.title
        existing.focus_type = focus_type or existing.focus_type
        existing.original_utterance = original_utterance or existing.original_utterance
        existing.corrected_utterance = corrected_utterance or existing.corrected_utterance
        existing.is_favorite = is_favorite
        existing.notes = notes or existing.notes
        db.commit()
        db.refresh(existing)
        return existing
    
    # Create new saved module
    saved_module = SavedModule(
        user_id=user.id,
        module_id=module_id,
        module_type=module_type,
        content=content,
        title=title,
        focus_type=focus_type,
        original_utterance=original_utterance,
        corrected_utterance=corrected_utterance,
        is_favorite=is_favorite,
        notes=notes
    )
    
    db.add(saved_module)
    db.commit()
    db.refresh(saved_module)
    return saved_module

def update_saved_module(
    db: Session,
    saved_module_id: int,
    is_favorite: Optional[bool] = None,
    notes: Optional[str] = None,
    practice_count: Optional[int] = None
) -> Optional[SavedModule]:
    """Update a saved module's user-specific fields."""
    saved_module = get_saved_module(db, saved_module_id)
    if not saved_module:
        return None
    
    if is_favorite is not None:
        saved_module.is_favorite = is_favorite
    
    if notes is not None:
        saved_module.notes = notes
    
    if practice_count is not None:
        saved_module.practice_count = practice_count
    
    db.commit()
    db.refresh(saved_module)
    return saved_module

def delete_saved_module(db: Session, saved_module_id: int) -> bool:
    """Delete a saved module."""
    saved_module = get_saved_module(db, saved_module_id)
    if not saved_module:
        return False
    
    db.delete(saved_module)
    db.commit()
    return True
