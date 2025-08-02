# IntelliFlow AI

> **Advanced AI workflow orchestration platform that intelligently combines document processing, real-time web search, and conversational AI to deliver contextual, accurate, and up-to-date responses.**

## 🌟 Overview

The **IntelliFlow AI** is a comprehensive AI-powered conversational system designed to bridge the gap between static document repositories and dynamic information needs. It combines advanced document processing, real-time web search, and intelligent conversation management to deliver contextual, accurate, and up-to-date responses to user queries.

### What Makes It Special?

- **🧠 Intelligent Intent Recognition**: Automatically determines when to search documents vs. the web
- **📚 Multi-Modal Document Support**: Processes PDFs, Word docs, PowerPoints, spreadsheets, and more
- **🔍 Hybrid Search Architecture**: Combines vector similarity search with advanced re-ranking
- **🌐 Real-Time Web Integration**: Fetches current information from the internet when needed
- **💬 Context-Aware Conversations**: Maintains conversation history with intelligent summarization
- **⚡ Asynchronous Processing**: Parallel execution for optimal performance
- **🚀 Production-Ready API**: FastAPI-based REST API with streaming support

## 🏗️ System Architecture

The system follows a **modular, node-based architecture** built on LangGraph for workflow orchestration:

### Core Workflow Components

#### 1. **Intent Detection Node**

- Analyzes user queries using keyword detection
- Determines whether document retrieval and/or web search is required
- Routes queries through appropriate processing pipelines

#### 2. **Query Decomposition Node**

- Breaks down complex queries into focused, answerable sub-tasks
- Handles multi-part questions intelligently
- Uses advanced prompt engineering for accurate task extraction

#### 3. **Retrieval Node**

- Performs vector similarity search across indexed documents
- Implements hybrid search with multiple embedding models
- Supports parallel retrieval for multiple query aspects

#### 4. **Search Node**

- Executes real-time web searches using Tavily API
- Extracts relevant content from search results
- Handles multiple search queries in parallel

#### 5. **Evaluation Node**

- Re-ranks and filters retrieved content for relevance
- Uses MS-Marco-MiniLM model for advanced re-ranking
- Combines and consolidates information from multiple sources

#### 6. **Aggregation Node**

- Merges document evidence with web search results
- Incorporates conversation history and summaries
- Prepares comprehensive context for response generation

#### 7. **Response Generation Node**

- Uses advanced language models (Llama-3.3-70B) for response generation
- Applies context-aware prompt templates
- Ensures responses are grounded in provided evidence

### Workflow Execution Flow

```mermaid
graph TD
    A[User Query] --> B[Intent Detection]
    B --> C{Needs Evidence?}
    C -->|No| H[Direct Response]
    C -->|Yes| D[Query Decomposition]
    D --> E{Retrieval + Search?}
    E -->|Both| F[Parallel Evidence Gathering]
    E -->|Documents Only| G[Document Retrieval]
    E -->|Web Only| I[Web Search]
    F --> J[Content Evaluation]
    G --> J
    I --> K[Context Aggregation]
    J --> K
    K --> L[Response Generation]
    H --> L
    L --> M[Final Response]
```

## 🛠️ Technology Stack

### Core AI & ML Frameworks

- **LangChain**: Comprehensive AI application framework
- **LangGraph**: Advanced workflow orchestration engine
- **Groq**: High-performance language model inference
- **FastEmbed**: Efficient embedding generation
- **Unstructured**: Advanced document parsing capabilities

### Language Models & Embeddings

- **Primary LLM**: Llama-3.3-70B (versatile, high-performance)
- **Utility LLM**: Llama3-8B (fast operations)
- **Embeddings**: BGE-Base-EN-v1.5, BGE-Small-EN-v1.5, Jina-v2-Small-EN
- **Re-ranking**: MS-Marco-MiniLM-L-12-v2

### Infrastructure & Storage

- **Vector Database**: Qdrant (high-performance vector similarity search)
- **Conversation Storage**: MongoDB (persistent state management)
- **Web Framework**: FastAPI (high-performance async API)
- **Search API**: Tavily (real-time web search)

### Document Processing Support

- **PDF**: Advanced parsing with layout preservation
- **Microsoft Office**: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **Text Files**: Markdown (.md), Plain Text (.txt)
- **Structured Data**: YAML (.yml), JSON

## 📋 Prerequisites & Setup

### Required API Services

1. **Groq API** (Language Models)

   - Sign up at: https://console.groq.com/
   - Get API key for Llama model access

2. **Tavily Search API** (Web Search)

   - Sign up at: https://app.tavily.com/
   - Get API key for real-time web search

3. **MongoDB** (Conversation Persistence)

   - Local installation OR MongoDB Atlas cloud service
   - Used for conversation state and checkpointing

4. **Qdrant** (Vector Database)
   - Local Docker installation OR Qdrant Cloud
   - Stores document embeddings for similarity search

### Environment Configuration

Create a `.env` file in the project root:

```env
# Groq API Configuration (Required)
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search API (Required)
# Get your API key from: https://app.tavily.com/
TAVILY_API_KEY=your_tavily_api_key_here

# MongoDB Configuration (Required)
# Local MongoDB instance
DB_URI=mongodb://localhost:27017
# Or use MongoDB Atlas (cloud)
# DB_URI=mongodb+srv://username:password@cluster.mongodb.net/doc_assistant

# Qdrant Configuration (Required)
# For local Qdrant instance (default)
QDRANT_URL=http://localhost:6333
# For Qdrant Cloud
# QDRANT_URL=https://your-cluster-url.qdrant.io
# QDRANT_API_KEY=your_qdrant_api_key

# Optional: DeepInfra API (alternative embedding provider)
# Get your token from: https://deepinfra.com/
# DEEPINFRA_API_TOKEN=your_deepinfra_token

# Optional: Custom cache directory for models
FASTEMBED_CACHE_PATH=./cache
```

## 🚀 Installation & Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/intelliflow-ai.git
cd intelliflow-ai
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Required Services

#### MongoDB Setup

```bash
# Option A: Local MongoDB with Docker
docker run -d --name mongodb -p 27017:27017 mongo:7

# Option B: Install MongoDB locally
# Download from: https://www.mongodb.com/try/download/community

# Option C: Use MongoDB Atlas (cloud)
# Sign up at: https://www.mongodb.com/atlas
```

#### Qdrant Setup

```bash
# Option A: Local Qdrant with Docker (Recommended)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Option B: Use Qdrant Cloud
# Sign up at: https://cloud.qdrant.io/
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your API keys and service URLs
# nano .env  # or use your preferred editor
```

### 5. Run the Application

#### Interactive Command Line Interface

```bash
python main.py
```

**CLI Commands:**

- Type any question to chat with the AI
- Add `"pdf"` or `"document"` keywords to trigger document search
- Add `"search"` or `"latest"` keywords to trigger web search
- Type `d` to delete current conversation thread
- Type `q` to quit

#### REST API Server

```bash
# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Access Points:**

- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API Reference

### Core Endpoints

#### Chat Operations

```http
# Send a chat message
POST /chat
Content-Type: application/json
{
    "user_id": "user123",
    "thread_id": "thread456",
    "user_message": "What does the document say about project timelines?",
    "use_documents": true,
    "use_internet": false
}

# Stream chat responses (real-time)
POST /chat/stream
Content-Type: application/json
{
    "user_id": "user123",
    "thread_id": "thread456",
    "user_message": "Search for latest AI developments",
    "use_documents": false,
    "use_internet": true
}

# Get conversation history
GET /chat_history/{user_id}

# Delete conversation thread
DELETE /delete_thread
Content-Type: application/json
{
    "user_id": "user123",
    "thread_id": "thread456"
}
```

#### Document Management

```http
# Upload and index documents
POST /upload_and_index
Content-Type: multipart/form-data
user_id: user123
thread_id: thread456
files: [file1.pdf, file2.docx, ...]

# Index document from URL or text
POST /index_attachment
Content-Type: application/json
{
    "user_id": "user123",
    "thread_id": "thread456",
    "file_url": "https://example.com/document.pdf"
}

# Delete user documents
DELETE /delete_user_document
Content-Type: application/json
{
    "user_id": "user123"
}
```

#### System Operations

```http
# Check system health
GET /health

# Delete collection (admin)
DELETE /collection
```

### Response Format

```json
{
  "user_id": "user123",
  "thread_id": "thread456",
  "bot_response": "Based on the provided document, the project timeline shows...",
  "success": true,
  "message": "Chat response generated successfully"
}
```

## ⚙️ Configuration & Customization

### Model Configuration (`config.py`)

```python
# Language Models
MODEL_ID = "llama-3.3-70b-versatile"      # Primary conversation model
UTILS_MODEL_ID = "llama3-8b-8192"         # Fast utility operations

# Embedding Models
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"  # Document embeddings
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"  # Content re-ranking

# Vector Database
QDRANT_HOST = "localhost"
QDRANT_PORT = 6334
QDRANT_COLLECTION = "attachments"

# Performance Tuning
RERANK_THRESHOLD = 0.1              # Re-ranking relevance threshold
SUMMARY_THRESHOLD = 8               # Messages before auto-summarization
MESSAGES_TO_RETAIN = 4              # Messages kept after summarization
```

### Advanced Customization

#### Intent Detection Keywords

Modify `services/intent_detection_node.py`:

```python
self.retrieval_keywords = ["pdf", "document", "doc", "file", "attachment", "paper"]
self.search_keywords = ["search", "latest", "updates", "recent", "web", "current"]
```

#### Query Decomposition Logic

Customize `services/decompose_node.py` to modify how complex queries are broken down.

#### Retrieval Parameters

Adjust search parameters in `services/retrieval_node.py`:

- Number of documents retrieved
- Similarity thresholds
- Embedding model selection

#### Prompt Templates

Modify response behavior in `config.py` by updating prompt templates for different scenarios.

## 🔄 Workflow Deep Dive

### Processing Pipeline

1. **Intent Analysis**

   ```
   User Query → Keyword Detection → Route Decision
   ├── Document Keywords Found → Enable Retrieval
   ├── Search Keywords Found → Enable Web Search
   └── No Keywords → Direct Response
   ```

2. **Query Decomposition** (if needed)

   ```
   Complex Query → LLM Analysis → Task Breakdown
   Example: "What's the budget and timeline?"
   → ["What is the budget?", "What is the timeline?"]
   ```

3. **Evidence Gathering** (parallel execution)

   ```
   Document Retrieval:           Web Search:
   ├── Vector Search            ├── Tavily API Call
   ├── Similarity Ranking       ├── Content Extraction
   └── Metadata Preservation    └── Source Attribution
   ```

4. **Content Evaluation & Re-ranking**

   ```
   Retrieved Content → Relevance Scoring → Content Consolidation
   ├── MS-Marco Re-ranking Model
   ├── Task-specific Grouping
   └── Duplicate Removal
   ```

5. **Context Aggregation**

   ```
   Sources Combined:
   ├── Conversation Summary
   ├── Document Evidence
   ├── Web Search Results
   └── User Query Context
   ```

6. **Response Generation**
   ```
   Aggregated Context → LLM Processing → Final Response
   ├── Context-aware Prompting
   ├── Source Attribution
   └── Accuracy Validation
   ```

### State Management

The system maintains conversation state through MongoDB checkpointing:

```python
State = {
    "user_query": str,              # Current user question
    "recent_messages": List,        # Conversation history
    "conversation_summary": str,     # Auto-generated summary
    "tasks": List[str],             # Decomposed query tasks
    "retrieved_docs": List,         # Document search results
    "web_search_results": List,     # Web search results
    "final_context": str,           # Aggregated context
    "do_retrieval": bool,           # Document search flag
    "do_search": bool,              # Web search flag
    "user_id": str,                 # User identifier
    "thread_id": str                # Conversation thread
}
```

## 🎯 Use Cases & Examples

### 1. Document Question Answering

```
User: "What are the key findings in the research paper? see pdf"
System:
├── Detects: Document retrieval needed
├── Searches: Indexed PDF documents
├── Finds: Relevant sections about findings
└── Responds: "According to the research paper, the key findings are..."
```

### 2. Current Information Lookup

```
User: "What are the latest developments in AI search technology?"
System:
├── Detects: Web search needed
├── Searches: Recent articles via Tavily
├── Aggregates: Current information
└── Responds: "Recent developments include..."
```

### 3. Hybrid Queries

```
User: "Compare the document's recommendations with current industry trends search"
System:
├── Detects: Both document + web search needed
├── Retrieves: Document recommendations (parallel)
├── Searches: Current industry trends (parallel)
├── Evaluates: Both sources for relevance
└── Responds: Comprehensive comparison
```

### 4. Multi-Document Analysis

```
User: "Summarize the budget information across all uploaded files"
System:
├── Searches: All indexed documents for budget info
├── Aggregates: Information from multiple sources
├── Consolidates: Findings by document source
└── Responds: Comprehensive budget summary
```

## 📊 Performance & Optimization

### System Performance

- **Parallel Processing**: Document retrieval and web search execute simultaneously
- **Async Operations**: Non-blocking I/O for optimal throughput
- **Caching**: Embedding model caching for faster startup
- **Connection Pooling**: Optimized database connections

### Scalability Features

- **Stateless Design**: API can be horizontally scaled
- **Vector Database**: Qdrant handles large document collections efficiently
- **Conversation Persistence**: MongoDB ensures conversation continuity
- **Background Processing**: Long-running tasks don't block user interactions

### Resource Management

```python
# Memory optimization
EMBEDDING_CACHE_SIZE = "1GB"        # Model cache limit
VECTOR_INDEX_MEMORY = "2GB"         # Qdrant memory allocation
MONGODB_POOL_SIZE = 10              # Connection pool size

# Processing limits
MAX_CONCURRENT_REQUESTS = 10        # API request limit
MAX_DOCUMENT_SIZE = "50MB"          # Per-document size limit
MAX_QUERY_LENGTH = 1000             # Character limit per query
```

## 🧪 Development & Testing

### Project Structure

```
intelliflow-ai/
├── 📄 api.py                          # FastAPI web server & endpoints
├── 📄 main.py                         # CLI interface & workflow runner
├── 📄 config.py                       # Configuration & prompts
├── 📄 requirements.txt                # Python dependencies
├── 📁 workflow/
│   └── 📄 graph.py                    # LangGraph workflow definition
├── 📁 services/                       # AI workflow nodes
│   ├── 📄 intent_detection_node.py    # Query intent analysis
│   ├── 📄 decompose_node.py           # Query task decomposition
│   ├── 📄 retrieval_node.py           # Document search & retrieval
│   ├── 📄 search_node.py              # Web search operations
│   ├── 📄 evaluater_node.py           # Content evaluation & re-ranking
│   ├── 📄 aggregation_node.py         # Context aggregation
│   ├── 📄 call_model.py               # Response generation
│   └── 📄 summarizer.py               # Conversation summarization
├── 📁 models/
│   └── 📄 state.py                    # State management schemas
├── 📁 utils/
│   └── 📄 checkpointer.py             # MongoDB persistence utilities
├── 📁 vector_db/                      # Document processing & vector ops
│   ├── 📄 document_indexer.py         # Document indexing pipeline
│   ├── 📄 document_processor.py       # File parsing & chunking
│   ├── 📄 embedding_manager.py        # Embedding generation
│   ├── 📄 qdrant_manager.py           # Vector database operations
│   ├── 📄 vector_service.py           # High-level vector services
│   └── 📄 example_usage.py            # Usage examples & testing
├── 📁 cache/                          # Model cache directory
└── 📄 .env.example                    # Environment configuration template
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/intelliflow-ai.git
cd intelliflow-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Additional dev tools

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Test basic functionality
python -c "from main import run_workflow; run_workflow()"

# Test API endpoints
pytest tests/ -v

# Test document processing
python vector_db/example_usage.py

# Test individual components
python services/decompose_node.py          # Test query decomposition
python services/retrieval_node.py          # Test document retrieval
python services/search_node.py             # Test web search
```

### Development Mode

```bash
# Start API with auto-reload
uvicorn api:app --reload --port 8000 --log-level debug

# Run CLI with debug logging
export LOG_LEVEL=DEBUG
python main.py

# Monitor system performance
# Add performance monitoring endpoints in api.py
```

### Code Quality & Standards

```bash
# Format code
black .

# Check code style
flake8 .

# Type checking
mypy .

# Run all checks
pre-commit run --all-files
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **API Key Errors**

```
Error: "Groq API key not found"
Solution: Ensure GROQ_API_KEY is set in .env file
```

#### 2. **Database Connection Issues**

```
Error: "MongoDB connection failed"
Solutions:
- Check DB_URI in .env file
- Ensure MongoDB service is running
- Verify network connectivity for cloud instances
```

#### 3. **Vector Database Issues**

```
Error: "Qdrant connection timeout"
Solutions:
- Verify Qdrant service is running on correct port
- Check QDRANT_URL configuration
- Ensure sufficient memory allocation
```

#### 4. **Document Processing Errors**

```
Error: "Document parsing failed"
Solutions:
- Check document file format support
- Verify file isn't corrupted
- Ensure sufficient disk space for processing
```

#### 5. **Memory Issues**

```
Error: "Out of memory during embedding generation"
Solutions:
- Reduce batch size in embedding generation
- Use smaller embedding models
- Increase system memory allocation
```

### Performance Optimization Tips

1. **Embedding Model Selection**

   - Use `bge-small-en-v1.5` for faster processing
   - Use `bge-base-en-v1.5` for better accuracy

2. **Document Chunking**

   - Adjust chunk size based on document type
   - Optimize overlap for better context preservation

3. **Vector Database Tuning**

   - Configure appropriate index parameters
   - Use quantization for memory efficiency

4. **API Response Time**
   - Enable response streaming for long queries
   - Implement request caching for repeated queries

## 🌐 Deployment

### Local Deployment

```bash
# Production server with Gunicorn
pip install gunicorn
gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables for Production

```env
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security settings
API_KEY_REQUIRED=true
CORS_ORIGINS=["https://yourdomain.com"]
MAX_REQUEST_SIZE=50MB

# Performance settings
WORKER_PROCESSES=4
CONNECTION_POOL_SIZE=20
CACHE_TTL=3600
```

### Health Monitoring

The `/health` endpoint provides system status:

```json
{
  "status": "healthy",
  "message": "Intelligent Document Assistant API is running",
  "workflow_initialized": true,
  "qdrant_status": "healthy",
  "mongodb_status": "connected",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

## 🤝 Contributing

We welcome contributions to improve IntelliFlow AI! Here's how to get started:

### Development Process

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with appropriate tests
4. **Follow code style guidelines** (Black, Flake8)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request** with detailed description

### Code Style Guidelines

- **PEP 8 compliance**: Use Black for formatting
- **Type hints**: Add type annotations for public methods
- **Docstrings**: Document all public classes and methods
- **Modular design**: Keep functions focused and testable
- **Error handling**: Use appropriate exception handling

### Areas for Contribution

- **🔌 New document formats**: Add support for additional file types
- **🧠 Advanced NLP**: Implement better query understanding
- **🚀 Performance**: Optimize vector search and embedding generation
- **📊 Analytics**: Add usage metrics and performance monitoring
- **🎨 UI/UX**: Create web interface for easier interaction
- **📱 Mobile**: Develop mobile-friendly API endpoints

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

**You are free to:**

- ✅ **Share** — copy and redistribute the material in any medium or format
- ✅ **Adapt** — remix, transform, and build upon the material

**Under the following terms:**

- 📝 **Attribution** — You must give appropriate credit and indicate if changes were made
- 🚫 **NonCommercial** — You may not use the material for commercial purposes

For the full license text, visit: https://creativecommons.org/licenses/by-nc/4.0/

## 🙏 Acknowledgments

Special thanks to the open-source community and the following projects that make this system possible:

- **🦜 LangChain Team** - Comprehensive AI application framework
- **⚡ Groq** - Lightning-fast language model inference
- **🔍 Qdrant** - High-performance vector database
- **🌐 Tavily** - Reliable web search API
- **🚀 FastAPI** - Modern, fast web framework
- **📊 MongoDB** - Flexible document database
- **🤗 Hugging Face** - Pre-trained embedding models
- **📄 Unstructured** - Advanced document processing

## 📞 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/MuneerRaza/intelliflow-ai/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/MuneerRaza/intelliflow-ai/discussions)
- **📖 Documentation**: API docs available at `/docs` when running the server
- **📧 Email**: [Author Email](muneer.raza098@gmail.com)

---

**Built with ❤️ for the AI community**

_Transforming how we interact with information through intelligent workflow orchestration_
