# Vector Database Test Script Usage

## Overview

`vec_test.py` is a comprehensive testing script for the OptimizedDocumentIndexer that allows you to test both "basic" and "hierarchical" retrieval strategies with your own files and queries.

## Quick Start

### 1. Configure Your Files and Queries

Open `vec_test.py` and modify the `main()` function:

```python
def main():
    # Add your file paths here
    file_paths = [
        r"C:\path\to\your\document1.pdf",
        r"C:\path\to\your\document2.txt",
        r"C:\path\to\your\document3.md",
    ]

    # Add your test queries here
    test_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Explain the main concepts",
    ]
```

### 2. Run the Test

```bash
python vec_test.py
```

## What the Script Does

### 🔄 **Comprehensive Testing**

- Tests both **basic** and **hierarchical** retrieval strategies
- Indexes your documents using local Qdrant storage
- Performs all your test queries on both strategies
- Compares performance metrics between strategies

### 📊 **Detailed Timing Logs**

- **Document loading time** for each file
- **Embedding calculation time** during indexing
- **Document upload time** to Qdrant
- **Query embedding time** during search
- **Search operation time** for each query
- **Total retrieval time** per query

### 📈 **Performance Comparison**

- Side-by-side comparison of both strategies
- Indexing time comparison
- Average search time comparison
- Winner determination for each metric
- Strategy recommendations based on results

### 🗂️ **Output Files**

- `vec_test.log` - Detailed execution logs
- `vec_test_results_[timestamp].json` - Complete test results in JSON format

## Example Output

```
======================================================================
TESTING BASIC STRATEGY
======================================================================
✅ Connected to local Qdrant storage at './qdrant' (connection time: 0.043s)
✅ Initialized embeddings with caching (initialization time: 4.487s)
📄 Document loading time for 'document1.pdf': 1.234s

--- Query 1/3: 'What is machine learning?' ---
🔍 Query embedding time: 0.156s
🔍 Search operation time: 0.089s
🔍 Total retrieval time: 0.245s (returned 3 results)

======================================================================
STRATEGY COMPARISON REPORT
======================================================================
📊 PERFORMANCE METRICS:
Metric                    Basic          Hierarchical   Winner
----------------------------------------------------------------------
Indexing Time            12.456         15.789         Basic
Avg Search Time          0.234          0.312          Basic
Total Time               45.123         52.456         Basic
```

## Supported File Types

The indexer supports various file formats:

- **Documents**: PDF, DOCX, DOC, ODT
- **Presentations**: PPTX, PPT
- **Spreadsheets**: XLSX, XLS, CSV
- **Web**: HTML, HTM
- **Markup**: MD (Markdown)
- **Code**: PY, JS, JAVA, CS, CPP, C, H, GO, RB, PHP, SWIFT, KT, TS, SQL, SH
- **Config**: YML, YAML, JSON
- **Text**: TXT

## Configuration Options

You can modify these parameters in the `test_strategy()` method:

```python
indexer = OptimizedDocumentIndexer(
    collection_name=f"test_collection_{strategy}",
    embedding_provider="fastembed",  # or "deepinfra"
    qdrant_path="./qdrant",          # Local storage path
    cache_path="./cache",            # Cache storage path
    strategy=strategy,               # "basic" or "hierarchical"
    chunk_size=800,                  # Text chunk size
    chunk_overlap=100,               # Overlap between chunks
    batch_size=50                    # Batch size for processing
)
```

## Strategy Comparison

### Basic Strategy

- ✅ **Faster** indexing and search
- ✅ **Consistent** chunk sizes
- ✅ **Lower** memory usage
- ✅ Good for **general-purpose** applications

### Hierarchical Strategy

- ✅ **Better** context preservation
- ✅ **More precise** search results
- ✅ Ideal for **complex** documents
- ✅ Better for **RAG** applications

## Quick Test with Project Files

If you want to quickly test with existing project files, uncomment this section in `main()`:

```python
import glob
file_paths = glob.glob("*.py") + glob.glob("*.md") + glob.glob("*.txt")
test_queries = [
    "What is this project about?",
    "How does the system work?",
    "What are the main components?",
]
```

## Troubleshooting

### File Not Found Errors

- Ensure file paths use raw strings: `r"C:\path\to\file.pdf"`
- Check that all files exist and are readable
- Use absolute paths for reliability

### Import Errors

- Ensure you're in the correct directory: `cd d:\Projects\AbundenceAISuite`
- Check that all required packages are installed
- Verify Python environment is activated

### Memory Issues

- Reduce `batch_size` in the configuration
- Process fewer files at once
- Use smaller `chunk_size` values

## Need Help?

Check the detailed logs in `vec_test.log` for debugging information.
