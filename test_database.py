"""
Test script for the database implementation.

This script tests the database functionality by creating a user,
saving a teaching module, retrieving it, updating it, and deleting it.
"""
import json
import os
from database.db import get_db_session, init_db
from database.crud import (
    get_or_create_user,
    save_module,
    get_user_saved_modules,
    update_saved_module,
    delete_saved_module
)

def test_database():
    """Run a simple test of the database functionality."""
    print("Initializing database...")
    init_db()
    
    # Test user operations
    user_id = "test_user_123"
    print(f"\nCreating user: {user_id}")
    
    with get_db_session() as db:
        user = get_or_create_user(db, user_id, username="Test User")
        print(f"User created/retrieved: {user}")
        
        # Load a sample teaching module from file
        module_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "teaching_module_generation",
            "output_files",
            "teaching_module_outputs_new",
            "validated"
        )
        
        # Find a module file
        module_files = os.listdir(module_dir)
        if not module_files:
            print("No teaching module files found. Please run the teaching module generation first.")
            return
        
        # Use the first module file
        module_file = os.path.join(module_dir, module_files[0])
        print(f"\nUsing teaching module: {module_file}")
        
        with open(module_file, 'r') as f:
            module_content = json.load(f)
        
        # Extract module information
        module_id = module_content.get("module_id", "unknown")
        module_type = module_content.get("module_type", "unknown")
        
        # Get source utterance info
        source_info = module_content.get("source_utterance_info", {})
        original_utterance = source_info.get("original", "")
        corrected_utterance = source_info.get("corrected", "")
        focus_type = source_info.get("focus_type", "")
        
        # Create a title
        title = f"Module: {focus_type} - {original_utterance[:30]}..."
        
        # Save the module
        print(f"\nSaving module: {module_id}")
        saved_module = save_module(
            db=db,
            user_id=user_id,
            module_id=module_id,
            module_type=module_type,
            content=module_content,
            title=title,
            focus_type=focus_type,
            original_utterance=original_utterance,
            corrected_utterance=corrected_utterance,
            is_favorite=False,
            notes="This is a test module."
        )
        print(f"Module saved with ID: {saved_module.id}")
        
        # Retrieve user's saved modules
        print("\nRetrieving user's saved modules:")
        user_modules = get_user_saved_modules(db, user_id)
        print(f"Found {len(user_modules)} modules:")
        for m in user_modules:
            print(f"  - ID: {m.id}, Module ID: {m.module_id}, Title: {m.title}")
        
        # Update the module
        print(f"\nUpdating module {saved_module.id} to mark as favorite")
        updated_module = update_saved_module(
            db=db,
            saved_module_id=saved_module.id,
            is_favorite=True,
            notes="This is an updated test module.",
            practice_count=1
        )
        print(f"Module updated: is_favorite={updated_module.is_favorite}, practice_count={updated_module.practice_count}")
        
        # Delete the module
        print(f"\nDeleting module {saved_module.id}")
        success = delete_saved_module(db, saved_module.id)
        print(f"Module deleted: {success}")
        
        # Verify deletion
        remaining_modules = get_user_saved_modules(db, user_id)
        print(f"Remaining modules: {len(remaining_modules)}")
    
    print("\nDatabase test completed successfully!")

if __name__ == "__main__":
    test_database()
