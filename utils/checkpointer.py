import os
import asyncio
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from dotenv import load_dotenv
load_dotenv()

# MongoDB connection with connection pooling and timeout
_client = None
_checkpointer = None

def get_mongo_client():
    """Get MongoDB client with proper connection pooling."""
    global _client
    if _client is None:
        db_uri = os.getenv("DB_URI", "mongodb://localhost:27017")
        _client = AsyncMongoClient(
            db_uri,
            maxPoolSize=10,  # Connection pool size
            minPoolSize=1,   # Minimum connections
            serverSelectionTimeoutMS=5000,  # 5s timeout for server selection
            connectTimeoutMS=10000,  # 10s timeout for connection
            socketTimeoutMS=5000,  # 5s timeout for socket operations
            retryWrites=True,  # Retry write operations
            retryReads=True    # Retry read operations
        )
    return _client

def get_checkpointer():
    """Get the checkpointer instance, creating it if necessary."""
    global _checkpointer
    if _checkpointer is None:
        client = get_mongo_client()
        _checkpointer = AsyncMongoDBSaver(client, db_name='abundance_ai')
    return _checkpointer

# For backward compatibility - will be created when first accessed
def checkpointer():
    return get_checkpointer()

async def delete_thread(user_id: str, thread_id: str):
    """Delete all checkpoint data for a specific user and thread."""
    try:
        client = get_mongo_client()
        db = client['abundance_ai']
        checkpoint = db['checkpoints']
        checkpoint_writes = db['checkpoint_writes']
        
        # Delete checkpoints for specific user and thread
        await checkpoint.delete_many({
            "user_id": user_id,
            "thread_id": thread_id
        })
        await checkpoint_writes.delete_many({
            "user_id": user_id,
            "thread_id": thread_id
        })
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection error: {e}")
        raise
    except Exception as e:
        print(f"Error deleting thread data: {e}")
        raise

async def delete_user_data(user_id: str):
    """Delete all checkpoint data for a specific user across all threads."""
    try:
        client = get_mongo_client()
        db = client['abundance_ai']
        checkpoint = db['checkpoints']
        checkpoint_writes = db['checkpoint_writes']
        
        # Delete all checkpoints for the user
        await checkpoint.delete_many({"user_id": user_id})
        await checkpoint_writes.delete_many({"user_id": user_id})
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection error: {e}")
        raise
    except Exception as e:
        print(f"Error deleting user data: {e}")
        raise

# Sync wrappers for backward compatibility
def delete_thread_sync(user_id: str, thread_id: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a task
            loop.create_task(delete_thread(user_id, thread_id))
        else:
            # If no event loop is running, we can use asyncio.run
            asyncio.run(delete_thread(user_id, thread_id))
    except RuntimeError:
        # Fallback: create a new event loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(delete_thread(user_id, thread_id))
        finally:
            new_loop.close()

def delete_user_data_sync(user_id: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a task
            loop.create_task(delete_user_data(user_id))
        else:
            # If no event loop is running, we can use asyncio.run
            asyncio.run(delete_user_data(user_id))
    except RuntimeError:
        # Fallback: create a new event loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(delete_user_data(user_id))
        finally:
            new_loop.close()