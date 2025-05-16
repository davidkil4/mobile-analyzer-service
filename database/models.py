"""
Database models for the analyzer service.

This module defines SQLAlchemy models for the database, ensuring compatibility
with both SQLite (development) and PostgreSQL (production).
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model for storing user-related data."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)  # External user ID (from auth system)
    username = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    saved_modules = relationship("SavedModule", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"


class SavedModule(Base):
    """Model for storing user-saved teaching modules."""
    __tablename__ = 'saved_modules'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    module_id = Column(String(100), nullable=False)  # Original module ID from the teaching module
    module_type = Column(String(50), nullable=False)  # e.g., "grammar", "vocabulary", "pattern"
    title = Column(String(200), nullable=True)
    focus_type = Column(String(50), nullable=True)  # e.g., "GRAMMAR", "VOCABULARY", "PATTERN"
    original_utterance = Column(Text, nullable=True)
    corrected_utterance = Column(Text, nullable=True)
    
    # Store the full module content as JSON
    # This allows flexibility while ensuring we can migrate to any DB that supports JSON
    content = Column(JSON, nullable=False)
    
    # Metadata
    is_favorite = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)  # User's personal notes about this module
    saved_at = Column(DateTime, default=datetime.utcnow)
    last_practiced_at = Column(DateTime, nullable=True)
    practice_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="saved_modules")
    
    def __repr__(self):
        return f"<SavedModule(id={self.id}, module_id='{self.module_id}', user_id={self.user_id})>"
