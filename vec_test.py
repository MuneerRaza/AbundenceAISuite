#!/usr/bin/env python3
"""
Vector Database Test Script
==========================

A comprehensive test script for the OptimizedDocumentIndexer that demonstrates:
- Document indexing with different strategies (basic, hierarchical)
- Performance timing and logging
- Batch processing capabilities
- Search functionality with detailed metrics
- Error handling and validation

Usage:
    1. Open this file in an editor
    2. Modify the 'file_paths' list in main() function with your document paths
    3. Modify the 'test_queries' list in main() function with your search queries
    4. Run: python vec_test.py

Example Configuration:
    file_paths = [
        r"C:\\Documents\\report1.pdf",
        r"C:\\Documents\\manual.txt", 
        r"C:\\Documents\\guide.md",
    ]
    
    test_queries = [
        "What is the main topic?",
        "How to install the software?",
        "What are the requirements?",
    ]

The script will test both 'basic' and 'hierarchical' strategies and provide
detailed performance comparisons and search results.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Literal

# Setup console encoding for Windows
if sys.platform == "win32":
    try:
        # Try to set console to UTF-8 mode
        os.system('chcp 65001 > nul')
    except:
        pass

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.vectordb import OptimizedDocumentIndexer


class VectorDatabaseTester:
    """
    Comprehensive tester for the OptimizedDocumentIndexer with detailed logging and metrics.
    """
    
    def __init__(self):
        # Setup logging for the test script with proper encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),  # Use stdout with proper encoding
                logging.FileHandler('vec_test.log', encoding='utf-8')  # Specify UTF-8 for log file
            ]
        )
        self.logger = logging.getLogger("VectorTester")
        self.test_results = []
        
    def validate_files(self, file_paths: List[str]) -> List[str]:
        """Validate that all provided file paths exist and are supported by the document loaders."""
        valid_files = []
        
        # Import the loader mapping from the OptimizedDocumentIndexer
        from services.vectordb import OptimizedDocumentIndexer
        supported_extensions = OptimizedDocumentIndexer.LOADER_MAPPING.keys()
        
        for file_path in file_paths:
            if os.path.isfile(file_path):
                file_ext = Path(file_path).suffix.lower()
                if file_ext in supported_extensions:
                    try:
                        # For binary files (like PDFs), just check if file exists and is readable
                        if file_ext in ['.pdf', '.docx', '.pptx', '.xlsx']:
                            # Just check file access without reading content
                            with open(file_path, 'rb') as f:
                                f.read(1)  # Try to read one byte
                        else:
                            # For text files, check UTF-8 readability
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(1)  # Try to read one character
                        
                        valid_files.append(file_path)
                        self.logger.info(f"Valid file: {file_path}")
                        print(f"[✓] Valid file: {Path(file_path).name}")
                    except Exception as e:
                        self.logger.warning(f"Cannot access file {file_path}: {e}")
                        print(f"[✗] Cannot access file {Path(file_path).name}: {e}")
                else:
                    self.logger.warning(f"Unsupported file type {file_ext}: {file_path}")
                    print(f"[✗] Unsupported file type {file_ext}: {Path(file_path).name}")
            else:
                self.logger.warning(f"File not found: {file_path}")
                print(f"[✗] File not found: {Path(file_path).name}")
        
        return valid_files
    
    def test_strategy(self, strategy: Literal["basic", "hierarchical"], files: List[str], thread_id: str, test_queries: List[str]) -> Dict[str, Any]:
        """Test a specific retrieval strategy with detailed metrics."""
        print(f"\n{'='*70}")
        print(f"TESTING {strategy.upper()} STRATEGY")
        print(f"{'='*70}")
        
        test_start_time = time.time()
        results = {
            "strategy": strategy,
            "files_count": len(files),
            "indexing_time": 0,
            "search_results": [],
            "errors": []
        }
        
        try:
            # Initialize indexer
            self.logger.info(f"Initializing {strategy} indexer...")
            indexer = OptimizedDocumentIndexer(
                collection_name=f"test_collection_{strategy}",
                embedding_provider="fastembed",  # Using fastembed for reliability
                qdrant_path="./qdrant",
                cache_path="./cache", 
                strategy=strategy,
                chunk_size=800,  # Moderate chunk size for testing
                chunk_overlap=100,
                batch_size=50  # Smaller batches for detailed logging
            )
            
            # Clean up before starting
            self.logger.info(f"Cleaning up before {strategy} test...")
            print(f"[i] Cleaning up collections and cache before {strategy} test...")
            indexer.delete_collection_if_exists()
            indexer.clear_cache()
            
            # Index files
            self.logger.info(f"Starting indexing of {len(files)} files...")
            indexing_start = time.time()
            indexer.index_files(files, thread_id=thread_id)
            results["indexing_time"] = time.time() - indexing_start
            
            print(f"\n[i] Testing {len(test_queries)} search queries...")
            for i, query in enumerate(test_queries, 1):
                print(f"\n--- Query {i}/{len(test_queries)}: '{query}' ---")
                
                search_start = time.time()
                search_results = indexer.search_documents(query, k=3, return_metadata=True)
                search_time = time.time() - search_start
                
                print(f"Search completed in {search_time:.3f}s - Found {len(search_results)} results")
                
                query_result = {
                    "query": query,
                    "search_time": search_time,
                    "results_count": len(search_results),
                    "results": []
                }
                
                for j, result in enumerate(search_results, 1):
                    print(f"\n  Result {j}:")
                    print(f"    Content Length: {result['content_length']} chars")
                    print(f"    Content Preview: {result['content'][:150]}...")
                    if 'source_file' in result.get('metadata', {}):
                        source_file = Path(result['metadata']['source_file']).name
                        print(f"    Source: {source_file}")
                    
                    query_result["results"].append({
                        "rank": result["rank"],
                        "content_length": result["content_length"],
                        "source_file": result.get("metadata", {}).get("source_file", "unknown")
                    })
                
                results["search_results"].append(query_result)
            
            # Cleanup
            print(f"\n[i] Cleaning up after {strategy} test...")
            indexer.delete_collection()
            self.logger.info(f"Cleaned up {strategy} test collection")
            
        except Exception as e:
            error_msg = f"Error in {strategy} strategy test: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            print(f"[✗] {error_msg}")
        
        results["total_time"] = time.time() - test_start_time
        return results
    
    def run_comprehensive_test(self, files: List[str], test_queries: List[str]):
        """Run comprehensive tests on both strategies."""
        if not files:
            self.logger.error("No files provided for testing")
            return
        
        if not test_queries:
            self.logger.error("No test queries provided")
            return
        
        print(f"\n[*] Starting comprehensive vector database test with {len(files)} files")
        print(f"Files to be indexed:")
        for file_path in files:
            print(f"  - {Path(file_path).name} ({Path(file_path).stat().st_size} bytes)")
        
        print(f"\nTest queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"  {i}. {query}")
        
        thread_id = f"test_session_{int(time.time())}"
        
        # Test both strategies
        strategies: List[Literal["basic", "hierarchical"]] = ["basic", "hierarchical"]
        all_results = []
        
        for strategy in strategies:
            result = self.test_strategy(strategy, files, thread_id, test_queries)
            all_results.append(result)
            self.test_results.append(result)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
    
    def generate_comparison_report(self, results: List[Dict[str, Any]]):
        """Generate a detailed comparison report between strategies."""
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON REPORT")
        print(f"{'='*70}")
        
        if len(results) != 2:
            print("[✗] Cannot generate comparison - need exactly 2 strategy results")
            return
        
        basic_result = next((r for r in results if r["strategy"] == "basic"), None)
        hierarchical_result = next((r for r in results if r["strategy"] == "hierarchical"), None)
        
        if not basic_result or not hierarchical_result:
            print("[✗] Missing strategy results for comparison")
            return
        
        print(f"\n[i] PERFORMANCE METRICS:")
        print(f"{'Metric':<25} {'Basic':<15} {'Hierarchical':<15} {'Winner':<10}")
        print("-" * 70)
        
        # Indexing time comparison
        basic_index_time = basic_result["indexing_time"]
        hier_index_time = hierarchical_result["indexing_time"]
        index_winner = "Basic" if basic_index_time < hier_index_time else "Hierarchical"
        print(f"{'Indexing Time':<25} {basic_index_time:<15.3f} {hier_index_time:<15.3f} {index_winner:<10}")
        
        # Average search time comparison
        if basic_result["search_results"] and hierarchical_result["search_results"]:
            basic_avg_search = sum(q["search_time"] for q in basic_result["search_results"]) / len(basic_result["search_results"])
            hier_avg_search = sum(q["search_time"] for q in hierarchical_result["search_results"]) / len(hierarchical_result["search_results"])
            search_winner = "Basic" if basic_avg_search < hier_avg_search else "Hierarchical"
            print(f"{'Avg Search Time':<25} {basic_avg_search:<15.3f} {hier_avg_search:<15.3f} {search_winner:<10}")
        else:
            print(f"{'Avg Search Time':<25} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
            basic_avg_search = hier_avg_search = 0
        
        # Total time comparison
        basic_total = basic_result["total_time"]
        hier_total = hierarchical_result["total_time"]
        total_winner = "Basic" if basic_total < hier_total else "Hierarchical"
        print(f"{'Total Time':<25} {basic_total:<15.3f} {hier_total:<15.3f} {total_winner:<10}")
        
        print(f"\n[i] STRATEGY RECOMMENDATIONS:")
        print("-" * 70)
        print("[*] BASIC Strategy:")
        print("   [+] Faster indexing and search")
        print("   [+] Consistent chunk sizes")
        print("   [+] Good for general-purpose applications")
        print("   [+] Lower memory usage")
        
        print("\n[*] HIERARCHICAL Strategy:")
        print("   [+] Better context preservation")
        print("   [+] More precise search results")
        print("   [+] Ideal for complex documents")
        print("   [+] Better for RAG applications")
        
        print(f"\n[!] CONCLUSION:")
        if basic_result["search_results"] and hierarchical_result["search_results"]:
            if basic_avg_search < hier_avg_search * 1.5:  # If basic is significantly faster
                print("   -> Use BASIC for speed-critical applications")
            else:
                print("   -> Use HIERARCHICAL for quality-critical applications")
        else:
            print("   -> No search results available for comparison")
        
        print("   -> Consider your specific use case and performance requirements")
        
        # Save results to file
        self.save_results_to_file(results)
    
    def save_results_to_file(self, results: List[Dict[str, Any]]):
        """Save test results to a JSON file."""
        import json
        
        output_file = f"vec_test_results_{int(time.time())}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[i] Test results saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


def main():
    """Main test execution function."""
    
    # ==========================================
    # CONFIGURATION: MODIFY THESE LISTS
    # ==========================================
    
    # Add your file paths here
    file_paths = [
        # Example file paths - replace with your actual files
        r"C:\Users\MuneerRaza\Downloads\CognifootAI_FYP_report_final.pdf",
        # r"C:\path\to\your\document2.txt",
        # r"C:\path\to\your\document3.md",
        # r"D:\Projects\AbundenceAISuite\README.md",
        # r"D:\Projects\AbundenceAISuite\config.py",
    ]
    
    # Add your test queries here
    test_queries = [
        # Example queries - replace with your actual queries
        "What is latent classifier?",
        # "How does artificial intelligence work?",
        # "Explain deep learning concepts",
        # "What are the main features?",
        # "How to configure the system?",
    ]
    
    # ==========================================
    # QUICK TEST WITH PROJECT FILES (UNCOMMENT TO USE)
    # ==========================================
    # If you want to quickly test with existing project files, uncomment these:
    
    # import glob
    # file_paths = glob.glob("*.py") + glob.glob("*.md") + glob.glob("*.txt")
    # test_queries = [
    #     "What is this project about?",
    #     "How does the system work?",
    #     "What are the main components?",
    # ]
    
    # ==========================================
    # END CONFIGURATION
    # ==========================================
    
    tester = VectorDatabaseTester()
    
    try:
        # Check if file paths and queries are provided
        if not file_paths:
            print("[✗] No file paths provided!")
            print("Please modify the 'file_paths' list in the main() function of vec_test.py")
            print("Example:")
            print('file_paths = [')
            print('    r"C:\\path\\to\\your\\document1.pdf",')
            print('    r"C:\\path\\to\\your\\document2.txt",')
            print(']')
            return
            
        if not test_queries:
            print("[✗] No test queries provided!")
            print("Please modify the 'test_queries' list in the main() function of vec_test.py")
            print("Example:")
            print('test_queries = [')
            print('    "What is machine learning?",')
            print('    "How does AI work?",')
            print(']')
            return
        
        # Validate files
        valid_files = tester.validate_files(file_paths)
        
        if not valid_files:
            print("[✗] No valid files found. Please check your file paths.")
            return
        
        print(f"\n[✓] Found {len(valid_files)} valid files out of {len(file_paths)} provided")
        
        # Run comprehensive test
        tester.run_comprehensive_test(valid_files, test_queries)
        
        print(f"\n[✓] Vector database testing completed successfully!")
        print(f"[i] Check 'vec_test.log' for detailed logs")
        
    except KeyboardInterrupt:
        print(f"\n[!] Test interrupted by user")
    except Exception as e:
        tester.logger.error(f"Test failed with error: {e}")
        print(f"[✗] Test failed: {e}")
    finally:
        print(f"\n[i] Thank you for testing the Vector Database system!")


if __name__ == "__main__":
    main()
