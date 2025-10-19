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

## 1. Initial Setup and Prerequisites

Before running any commands, ensure your environment is configured correctly.

### 1.1. Environment and Models

1. **Ollama:** Ensure the **Ollama server is running** in the background.
    
2. **Models:** The system requires two models, which must be pulled via Ollama:
    
    - **LLM (for Generation):** `llama3.2:latest` (or the model configured in `rag_agentic.py`)
        
    - **Embedder (for Vectorization):** `mxbai-embed-large:335m` (or the model configured in `rag_embedder.py`)
        

### 1.2. Configuration Check

Verify the settings in your `config.py` file, especially the paths:

|   |   |   |
|---|---|---|
|**Variable**|**Current Value (from config.py)**|**Description**|
|`MASTER_DOCS_PATH`|`D:\obsidian notes\Note`|The folder containing your source files (.pdf, .md).|
|`CHROMA_DB_PATH`|`D:\rag_storage`|Where the vector database files are stored.|
|`EMBEDDING_API_BATCH_SIZE`|`1000`|Batches size for embedding (optimized for speed).|

## 2. Data Management (Using `main.py`)

The `main.py` script is your command-line tool for preparing and managing the vector database. **You MUST run the `index` mode at least once before using the `query` or `app` modes.**

### 2.1. 📥 Index Mode (Initial Ingestion and Updates)

This is the most critical step. It parses, chunks, embeds, and indexes your documents into ChromaDB. It only re-indexes files that have been modified since the last run.

|   |   |
|---|---|
|**Command**|**Purpose**|
|**`python main.py --mode index --folder "D:\obsidian notes\Note"`**|Scans the folder, checks for new/modified documents, deletes old chunks, and adds new vectors.|

### 2.2. 🔎 Query Mode (CLI Test)

Run a direct query against your RAG system to confirm the pipeline is working correctly.

|   |   |
|---|---|
|**Command**|**Purpose**|
|**`python main.py --mode query --query "Explain the concept of margin of safety from the Deep Value book."`**|Executes the full RAG pipeline (embed query, retrieve, re-rank, generate answer).|

### 2.3. 🗑️ Wipe Mode (Database Reset)

Use this command to completely and permanently delete **all** documents from the `rag_docs` collection in your ChromaDB.

|   |   |
|---|---|
|**Command**|**Purpose**|
|**`python main.py --mode wipe`**|Clears the database. **You must type 'yes' to confirm deletion.**|

## 3. Conversational Interface (Using `app.py`)

The Streamlit application provides a user-friendly, persistent, and feature-rich interface for interacting with your RAG system.

### 3.1. 🌐 Launch the Application

Launch the web interface using the following command:

|   |   |
|---|---|
|**Command**|**Purpose**|
|**`streamlit run app.py`**|Starts the web server (usually on port 8501).|

### 3.2. ✨ Key Web Features

|   |   |   |
|---|---|---|
|**Feature**|**How to Use**|**Benefit**|
|**Persistence**|Automatically saves and loads chat history from `chat_persistence.json`.|You never lose your conversation log across sessions.|
|**RAG Mode**|**Default mode.** Automatically retrieves documents for every query.|Provides grounded, sourced answers from your private knowledge base.|
|**Chat Mode**|Type the command **`/chat`** into the input box.|Bypasses the database to allow for general conversation using the LLM.|
|**Switch Back**|Type the command **`/rag`** into the input box.|Re-enables document retrieval for grounded answers.|
|**Context Display**|Click the **"Show Context & Sources"** expander under the Bot's answer.|View the exact document chunks and filenames used to formulate the response.|

