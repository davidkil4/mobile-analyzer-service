"""
Database connection and utility functions.

This module provides database connection management and utility functions
that abstract the underlying database implementation (SQLite/PostgreSQL).
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from pathlib import Path

from database.models import Base

logger = logging.getLogger(__name__)

# Default to SQLite for development
DEFAULT_DB_URL = "sqlite:///database/teaching_modules.db"

# Environment variable to control database connection
DB_URL = os.environ.get("DB_URL", DEFAULT_DB_URL)

# Create database directory if it doesn't exist
if DB_URL.startswith("sqlite"):
    db_path = DB_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

# Create engine based on DB_URL
if DB_URL.startswith("sqlite"):
    # SQLite-specific settings
    engine = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},  # Allow multi-threaded access
        echo=False,  # Set to True for SQL query logging
    )
else:
    # PostgreSQL or other database settings
    engine = create_engine(
        DB_URL,
        pool_size=5,  # Connection pool size
        max_overflow=10,  # Max extra connections when pool is full
        pool_timeout=30,  # Seconds to wait for connection from pool
        pool_recycle=1800,  # Recycle connections after 30 minutes
        echo=False,  # Set to True for SQL query logging
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# For async operations (when using PostgreSQL with asyncpg)
if not DB_URL.startswith("sqlite") and "postgresql" in DB_URL:
    # Convert to async URL by replacing postgresql:// with postgresql+asyncpg://
    async_db_url = DB_URL.replace("postgresql://", "postgresql+asyncpg://")
    async_engine = create_async_engine(async_db_url)
    AsyncSessionLocal = sessionmaker(
        class_=AsyncSession, autocommit=False, autoflush=False, bind=async_engine
    )


def init_db():
    """Initialize the database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


@contextmanager
def get_db_session():
    """Get a database session using a context manager.
    
    Usage:
        with get_db_session() as session:
            user = session.query(User).filter(User.id == 1).first()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


async def get_async_db_session():
    """Get an async database session for use with FastAPI dependency injection.
    
    This is for PostgreSQL with asyncpg only. For SQLite, use get_db_session.
    
    Usage:
        @app.get("/items/")
        async def read_items(session: AsyncSession = Depends(get_async_db_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
    """
    if DB_URL.startswith("sqlite"):
        raise ValueError("Async sessions are not supported with SQLite")
        
    async_session = AsyncSessionLocal()
    try:
        yield async_session
        await async_session.commit()
    except Exception as e:
        await async_session.rollback()
        logger.error(f"Async database session error: {e}")
        raise
    finally:
        await async_session.close()


def is_postgres():
    """Check if we're using PostgreSQL."""
    return not DB_URL.startswith("sqlite")
