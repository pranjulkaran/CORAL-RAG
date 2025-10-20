# 🚀 Agentic RAG System Execution Guide

This guide outlines the steps and commands required to run your custom Retrieval-Augmented Generation (RAG) system, which uses Ollama, ChromaDB, and a two-stage agentic pipeline.

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

This is the most critical step. It parses, chunks, embeds, and indexes your documents into ChromaDB. It only re-indexes files that have been modified (or moved) since the last run, leveraging **Content Hash** and **Scoped Cleanup** for efficiency and safety.

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

### 2.4. 🔎 app Mode (RUN command)

Run a direct run command using main

|   |   |
|---|---|
|**Command**|**Purpose**|
|**`python main.py --mode app `**|Executes the full RAG pipeline (embed query, retrieve, re-rank, generate answer using the app.py streamlit ui local host).|


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
|**Context Display**|Click the **"View Re-Ranked Chunks"** expander under the Bot's answer.|View the exact document chunks and filenames used to formulate the response.|
|**Source Citation**|Sources are displayed as **"Cited Sources"** badges.|Provides verifiable document names for the information retrieved.|
|**Start New Chat**|Click the **"Start New Chat"** button in the sidebar.|Clears the history so the LLM doesn't incorrectly use old context in a new conversation.|

|**Parameters**|View the sidebar for settings like **`top_k_retrieve`** and **`top_n_rank`**.|Gives transparency into the RAG agent's retrieval configuration.|
