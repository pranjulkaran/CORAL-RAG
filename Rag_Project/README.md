# 🧠 Incremental RAG System: Local Knowledge Base & Memory-Efficient Pipeline

## 🌟 Project Overview

This system implements a highly optimized, local **Retrieval-Augmented Generation (RAG)** solution designed for large, frequently updated knowledge bases. It uses an all-local stack (**Ollama** for LLMs/Embeddings and **ChromaDB** for vector storage) to ensure privacy, speed, and cost-effectiveness.

The core strength of this project is its **Incremental Ingestion Pipeline**, which dramatically cuts down update time by intelligently skipping redundant work and managing memory usage for massive datasets.

## ✨ Core Features & Efficiency Innovations

|   |   |   |
|---|---|---|
|**Category**|**Feature**|**Description**|
|**🚀 Ingestion Speed**|**Incremental Caching**|Uses a two-step process: **File-Level Skip** (`mtime` check) and **Chunk-Level Hashing** check against ChromaDB. The system only sends chunks with genuinely new content to the Ollama embedder.|
|**💾 Memory Management**|**Batched Indexing**|Implements the **`EMBEDDING_INDEXING_BATCH_SIZE = 1000`** strategy. Chunks are embedded and indexed sequentially in batches of 1,000, preventing RAM overflow and addressing ChromaDB's internal batch limits.|
|**🌐 Local Stack**|**Ollama & ChromaDB**|All processing runs on your local machine, guaranteeing data privacy and low-latency performance using models like `mxbai-embed-large` and `llama3.2:latest`.|
|**🧹 Data Integrity**|**Stale Chunk Cleanup**|Automatically identifies and removes vectors associated with source files that have been deleted or moved from the disk.|
|**💻 Interface**|**Streamlit UI**|Provides an interactive chat experience with crucial transparency features like Source Citation and Raw Context Visualization.|

## 🏗️ Architecture Deep Dive

The system operates using a two-part asynchronous architecture: the Ingestion Engine and the RAG Chat Agent.

### 1. The Ingestion Engine (`ingestion_pipeline.py`)

This component is the intelligence behind the incremental updates. It manages the synchronization between your source files and the ChromaDB collection.

|   |   |   |
|---|---|---|
|**Stage**|**Mechanism in Detail**|**Efficiency Gain**|
|**File Change Detection**|Compares the file's current **Modification Time (`mtime`)** with the `file_mtime` metadata stored in the database.|**Fastest Skip:** If the timestamps match, the entire file is skipped, saving seconds or minutes per document.|
|**Chunk-Level Caching**|Before embedding, the content of the chunk is hashed (`hashlib.sha256`) to create a unique ID. This ID is checked against the database.|**Resource Saving:** If the ID exists, the chunk's vector is assumed to be identical and is **skipped from the Ollama API call**, reducing compute time and cost.|
|**Memory-Safe Indexing**|The system iterates over the list of new chunks in batches of **1,000** (`EMBEDDING_INDEXING_BATCH_SIZE`). It calls `await self.embedder.embed_batch()` and then `self.collection.add()` sequentially for each batch.|**Stability:** Avoids peak memory spikes and prevents `ValueError: Batch size... is greater than max batch size` errors from ChromaDB.|
|**Chunking Stability**|Uses a conservative `chunk_size=512`.|**Reliability:** Prevents the Ollama error: `the input length exceeds the context length (status code: 500)`, ensuring even dense PDF files are processed.|

### 2. The RAG Chat Agent (`rag_agentic.py` - _Assumed_)

This component handles real-time querying against the indexed vector store (ChromaDB).

|   |   |   |
|---|---|---|
|**Stage**|**Functionality**|**Outcome**|
|**Retrieval**|Uses the user query to search the vector database. It then typically implements a two-stage process (Candidate Generation and Re-Ranking) to pull the most relevant **Top N** chunks.|Reduces noisy context, ensuring the LLM receives the highest quality, most relevant data.|
|**Generation**|Passes the curated context, the user query, and the chat history to the Ollama `llama3.2:latest` model, guided by a system prompt.|Generates a grounded, comprehensive answer that is strictly based on the provided documents.|

## 🛠️ Prerequisites and Setup

### Prerequisites

Ensure the following software and models are installed and running on your system:

1. **Python 3.9+**
    
2. **Ollama:** Must be installed and running locally.
    
3. **Required Ollama Models:** Pull the two models used by the project:
    
    ```
    ollama pull mxbai-embed-large
    ollama pull llama3.2:latest
    ```
    
4. **Source Documents:** Organize all your `.pdf` and `.md` files in a dedicated, absolute path folder.
    

### Installation

1. Install Python dependencies:
    
    (You may need to add additional dependencies like python-dotenv if not already installed)
    
    ```
    pip install chromadb ollama rich streamlit langchain-text-splitters langchain-community
    ```
