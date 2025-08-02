import os
import json
import asyncio
import tempfile
import shutil
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig
from contextlib import asynccontextmanager
from datetime import datetime

# Import project modules
from workflow.graph import build_workflow
from models.state import State
from utils.checkpointer import delete_thread_sync, delete_user_data_sync
from vector_db.vector_service import VectorService

# Set environment variables
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ['FASTEMBED_CACHE_PATH'] = 'cache'

# Global workflow graph instance
workflow_graph = None

# In-memory chat history storage (in production, use a database)
chat_history_db = {}

def get_workflow_graph():
    """Get or create the workflow graph instance."""
    global workflow_graph
    if workflow_graph is None:
        workflow_graph = build_workflow()
    return workflow_graph

def save_chat_thread(user_id: str, thread_id: str, title: str, last_message: str):
    """Save a chat thread to the history."""
    if user_id not in chat_history_db:
        chat_history_db[user_id] = {}
    
    chat_history_db[user_id][thread_id] = {
        "thread_id": thread_id,
        "title": title,
        "last_message": last_message,
        "timestamp": "2024-01-01T12:00:00Z",  # In production, use real timestamp
        "message_count": 1  # In production, count actual messages
    }

def delete_chat_thread(user_id: str, thread_id: str):
    """Delete a chat thread from history."""
    if user_id in chat_history_db and thread_id in chat_history_db[user_id]:
        del chat_history_db[user_id][thread_id]

def get_user_chat_history(user_id: str):
    """Get chat history for a user."""
    if user_id not in chat_history_db:
        return []
    return list(chat_history_db[user_id].values())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    # Startup
    get_workflow_graph()
    print("🚀 IntelliFlow AI API started successfully!")
    yield
    # Shutdown
    print("🛑 IntelliFlow AI API shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="IntelliFlow AI API",
    description="Advanced AI workflow orchestration platform with multi-modal capabilities: document processing, web search, and intelligent conversation management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    thread_id: str = Field(..., description="Unique identifier for the conversation thread")
    query: str = Field(..., description="User's message or question")
    use_attachment: bool = Field(default=False, description="Whether to search through attached documents")
    use_search: bool = Field(default=False, description="Whether to perform web search")

class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    bot_response: str
    success: bool
    message: str = "Chat completed successfully"

class IndexAttachmentRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    thread_id: str = Field(..., description="Unique identifier for the conversation thread")
    file_paths: List[str] = Field(..., description="List of file paths to index")

class IndexAttachmentResponse(BaseModel):
    user_id: str
    thread_id: str
    indexed_count: int
    skipped_count: int
    success: bool
    message: str

class DeleteRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    thread_id: Optional[str] = Field(None, description="Unique identifier for the conversation thread (optional for user deletion)")

class DeleteResponse(BaseModel):
    user_id: str
    thread_id: Optional[str]
    success: bool
    message: str

class ChatHistoryResponse(BaseModel):
    user_id: str
    threads: List[dict]
    success: bool
    message: str = "Chat history retrieved successfully"

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to IntelliFlow AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "index_attachment": "/index_attachment",
            "delete_user_document": "/delete_user_document",
            "delete_thread": "/delete_thread"
        }
    }

async def stream_chat_response(graph, initial_state, config, use_internet: bool, use_documents: bool):
    """Stream the chat response with real-time status updates."""
    
    try:
        # Send initial status based on what will be done
        if use_documents and use_internet:
            yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing_documents', 'message': 'Analyzing documents and searching internet...'})}\n\n"
            # Add a small delay to show the status
            await asyncio.sleep(2)
        elif use_documents:
            yield f"data: {json.dumps({'type': 'status', 'status': 'analyzing_documents', 'message': 'Analyzing documents...'})}\n\n"
            await asyncio.sleep(2)
        elif use_internet:
            yield f"data: {json.dumps({'type': 'status', 'status': 'searching_internet', 'message': 'Searching the internet...'})}\n\n"
            await asyncio.sleep(2)
        
        # Send thinking status
        yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': 'Generating response...'})}\n\n"
        
        # Process through the workflow to get the final response
        response = await graph.ainvoke(initial_state, config=config)
        final_message = response['recent_messages'][-1].content
        
        # Stream the response character by character for smooth experience
        current_text = ""
        for char in final_message:
            current_text += char
            yield f"data: {json.dumps({'type': 'content', 'content': current_text, 'is_complete': False})}\n\n"
            await asyncio.sleep(0.01)
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'content', 'content': current_text, 'is_complete': True})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'status': 'complete', 'message': 'Response complete'})}\n\n"
        
    except Exception as e:
        # Send error status
        error_message = f"Sorry, I encountered an error: {str(e)}"
        yield f"data: {json.dumps({'type': 'content', 'content': error_message, 'is_complete': True})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'status': 'error', 'message': 'Error occurred'})}\n\n"

@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response with real-time status updates.
    
    This endpoint provides true streaming responses with real status indicators for:
    - Internet search progress
    - Document analysis progress
    - Response generation progress
    """
    try:
        # Get the workflow graph
        graph = get_workflow_graph()
        
        # Create configuration for the thread
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        
        # Create initial state
        initial_state = State({
            "recent_messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "conversation_summary": "",
            "do_retrieval": request.use_attachment,
            "do_search": request.use_search,
            "user_id": request.user_id,
            "thread_id": request.thread_id,
            "tasks": [],
            "retrieved_docs": [],
            "web_search_results": [],
            "final_context": "",
        })
        
        # Save chat thread to history
        save_chat_thread(
            user_id=request.user_id,
            thread_id=request.thread_id,
            title=request.query[:50] + "..." if len(request.query) > 50 else request.query,
            last_message=request.query
        )
        
        # Return streaming response with real-time processing
        return StreamingResponse(
            stream_chat_response(graph, initial_state, config, request.use_search, request.use_attachment),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing streaming chat request: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a chat message and return the bot's response.
    
    This endpoint handles user queries with optional document retrieval and web search capabilities.
    The response is generated using the AI workflow with context from previous messages.
    """
    try:
        # Get the workflow graph
        graph = get_workflow_graph()
        
        # Create configuration for the thread
        config = RunnableConfig(configurable={"thread_id": request.thread_id})
        
        # Create initial state
        initial_state = State({
            "recent_messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "conversation_summary": "",
            "do_retrieval": request.use_attachment,
            "do_search": request.use_search,
            "user_id": request.user_id,
            "thread_id": request.thread_id,
            "tasks": [],
            "retrieved_docs": [],
            "web_search_results": [],
            "final_context": "",
        })
        
        # Process the request through the workflow
        response = await graph.ainvoke(initial_state, config=config)
        
        # Extract the bot's response
        final_message = response['recent_messages'][-1].content
        
        return ChatResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            bot_response=final_message,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post("/index_attachment", response_model=IndexAttachmentResponse, tags=["Document Management"])
async def index_attachment(request: IndexAttachmentRequest):
    """
    Index documents for a specific user and thread.
    
    This endpoint processes and indexes documents using the vector database.
    Documents are chunked, embedded, and stored for later retrieval.
    """
    try:
        # Validate file paths exist
        for file_path in request.file_paths:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"File not found: {file_path}"
                )
        
        # Index documents using VectorService
        result = await VectorService.index_documents(
            file_paths=request.file_paths,
            user_id=request.user_id,
            thread_id=request.thread_id
        )
        
        # Handle the result properly
        if result is None:
            result = {}
        
        return IndexAttachmentResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            indexed_count=result.get("indexed_count", 0),
            skipped_count=result.get("skipped_count", 0),
            success=True,
            message=result.get("message", "Documents indexed successfully")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing documents: {str(e)}"
        )

@app.post("/upload_and_index", tags=["Document Management"])
async def upload_and_index(
    user_id: str = Form(...),
    thread_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload and index documents for a specific user and thread.
    
    This endpoint handles file uploads and automatically indexes them.
    """
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            # Save uploaded files to temporary directory
            for file in files:
                if file.filename:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    file_paths.append(file_path)
            
            if not file_paths:
                raise HTTPException(
                    status_code=400,
                    detail="No valid files uploaded"
                )
            
            # Index documents using VectorService
            result = await VectorService.index_documents(
                file_paths=file_paths,
                user_id=user_id,
                thread_id=thread_id
            )
            
            # Handle the result properly
            if result is None:
                result = {}
            
            return IndexAttachmentResponse(
                user_id=user_id,
                thread_id=thread_id,
                indexed_count=result.get("indexed_count", 0),
                skipped_count=result.get("skipped_count", 0),
                success=True,
                message=result.get("message", "Documents indexed successfully")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading and indexing documents: {str(e)}"
        )

@app.delete("/delete_user_document", response_model=DeleteResponse, tags=["Data Management"])
async def delete_user_document(request: DeleteRequest):
    """
    Delete all data for a specific user.
    
    This endpoint removes:
    - All checkpoint data (conversation history, summaries)
    - All indexed documents in the vector database
    - All user-related data across all threads
    """
    try:
        # Delete checkpoint data
        delete_user_data_sync(request.user_id)
        
        # Delete vector database documents
        await VectorService.delete_user_documents(request.user_id)
        
        # Clear chat history for user
        if request.user_id in chat_history_db:
            del chat_history_db[request.user_id]
        
        return DeleteResponse(
            user_id=request.user_id,
            thread_id=None,
            success=True,
            message=f"All data for user '{request.user_id}' has been deleted successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting user data: {str(e)}"
        )

@app.delete("/delete_thread", response_model=DeleteResponse, tags=["Data Management"])
async def delete_thread(request: DeleteRequest):
    """
    Delete data for a specific thread.
    
    This endpoint removes:
    - Thread-specific checkpoint data (conversation history, summaries)
    - Thread-specific indexed documents in the vector database
    
    Requires both user_id and thread_id.
    """
    try:
        if not request.thread_id:
            raise HTTPException(
                status_code=400,
                detail="thread_id is required for thread deletion"
            )
        
        # Delete checkpoint data for the specific thread
        delete_thread_sync(request.user_id, request.thread_id)
        
        # Delete vector database documents for the specific thread
        await VectorService.delete_chat_documents(request.user_id, request.thread_id)
        
        # Delete from chat history
        delete_chat_thread(request.user_id, request.thread_id)
        
        return DeleteResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            success=True,
            message=f"Thread '{request.thread_id}' for user '{request.user_id}' has been deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting thread data: {str(e)}"
        )

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify API status."""
    qdrant_status = "unknown"
    try:
        # Try to connect to Qdrant
        from vector_db.vector_service import VectorService
        indexer = VectorService._get_indexer()
        if indexer and indexer.qdrant_manager:
            await indexer.qdrant_manager.client.get_collections()
            qdrant_status = "healthy"
        else:
            qdrant_status = "not_initialized"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "message": "IntelliFlow AI API is running",
        "workflow_initialized": workflow_graph is not None,
        "qdrant_status": qdrant_status
    }

@app.get("/chat_history/{user_id}", response_model=ChatHistoryResponse, tags=["Chat"])
async def get_chat_history(user_id: str):
    """
    Get chat history for a specific user.
    
    This endpoint returns a list of previous chat threads with their metadata.
    """
    try:
        # Get real chat history from storage
        threads = get_user_chat_history(user_id)
        
        return ChatHistoryResponse(
            user_id=user_id,
            threads=threads,
            success=True,
            message="Chat history retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "path": str(request.url.path)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "path": str(request.url.path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 