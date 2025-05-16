"""
Database package for analyzer service.

This package provides database models, connections, and API endpoints
for storing user-saved teaching modules.
"""
from database.db import init_db, get_db_session, is_postgres
from database.models import User, SavedModule
from database.crud import (
    get_user, create_user, get_or_create_user,
    get_saved_module, get_user_saved_modules,
    save_module, update_saved_module, delete_saved_module
)

__all__ = [
    'init_db', 'get_db_session', 'is_postgres',
    'User', 'SavedModule',
    'get_user', 'create_user', 'get_or_create_user',
    'get_saved_module', 'get_user_saved_modules',
    'save_module', 'update_saved_module', 'delete_saved_module'
]
