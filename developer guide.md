# 🛠️ Developer Guide: Agentic RAG Pipeline

This document is for developers who need to understand, debug, or extend the functionality of the custom RAG pipeline. The architecture is designed for **modularity**, **efficiency**, and **SOTA retrieval performance**.

## 1. System Design Philosophy

The system prioritizes efficiency and safety by externalizing expensive operations and providing defensive coding for data management:

- **Asynchronous Embedding:** All embedding calls are non-blocking (async) to maximize throughput to the Ollama API, reducing ingestion time.
    
- **Defensive Indexing:** Uses **Content Hashing** to prevent costly re-embedding of moved files and **Scoped Cleanup** to ensure indexing one folder does not delete data from another.
    
- **Modularity:** Each major component (Embedding, Vector DB, Ingestion, Agent) is isolated in its own file for easy swapping (e.g., replacing ChromaDB with Pinecone or the Ollama LLM with a Gemini API call).
    

## 2. Technical File Reference

|   |   |   |
|---|---|---|
|**File**|**Purpose and Responsibilities**|**Key Components & Logic**|
|**`ingestion_pipeline.py`**|**The Core ETL Engine.** Responsible for the entire data preparation pipeline: file scanning, loading, chunking, and incremental update logic.|`IngestPipeline` Class: `parse_docs` (Move Detection, Mtime Check, Hashing). `index_docs` (Batching, Deduplication, UPSERT). `cleanup_deleted_files` (Scoped Cleanup logic).|
|**`rag_embedder.py`**|**Embedding Service Connector.** Handles the communication with the local Ollama server specifically for the embedding model (`mxbai-embed-large:335m`).|`OllamaBatchEmbedder` Class: Contains the asynchronous `embed_batch` method to generate vectors for lists of text chunks efficiently.|
|**`vector_db_factory.py`**|**Database Abstraction.** Isolates the initialization and connection logic for the vector database (ChromaDB).|`get_vector_db` Function: Ensures a single, configured instance of the ChromaDB collection is returned globally.|
|**`rag_agentic.py`**|**The Query Orchestrator.** Manages the entire online query process: embedding the user question, performing RAG retrieval, re-ranking, and generating the final LLM response.|`AgenticRAG` Class: `query` method (The full RAG execution logic). Handles prompt construction and the two-stage retrieval process.|
|**`main.py`**|**Command-Line Interface (CLI).** The application entry point for developer and administrator tasks (indexing, query testing, wiping the database).|Handles command-line argument parsing (`--mode`, `--folder`, etc.) and calls the primary methods in `ingestion_pipeline.py`.|
|**`streamlit_ui.py`**|**Frontend Application.** The conversational interface for the end-user. Handles session state, displays chat history, and visualizes the context chunks and sources.|Manages Streamlit components (`st.chat_message`, `st.sidebar`). **Crucially, it handles the `new_chat` functionality to reset history.**|

## 3. Extension and Modification Guide

Developers may need to swap out components to test new models or storage options.

### 3.1. Swapping the LLM (Generator)

To use a different Large Language Model for generation (e.g., swapping `llama3.2:latest` for a larger model or a commercial API):

1. **Locate:** Update the model name reference inside **`rag_agentic.py`**.
    
2. **Logic Change:** If switching to a non-Ollama model (e.g., Gemini API or OpenAI), you must modify the `rag_agentic.py`'s `query` method to use the appropriate API client and connection logic instead of the current Ollama client wrapper.
    

### 3.2. Swapping the Embedder

To replace the `mxbai-embed-large` model:

1. **Locate:** Update the model name reference inside **`rag_embedder.py`**.
    
2. **Logic Change:** If switching to a commercial embedding service (e.g., Google or Cohere), the `embed_batch` method in **`rag_embedder.py`** must be entirely rewritten to use that service's SDK.
    

### 3.3. Changing Chunking Strategy

To adjust how documents are broken down:

1. **Locate:** The `RecursiveCharacterTextSplitter` initialization in **`ingestion_pipeline.py`** within the `index_docs` method.
    
2. **Modify:** Change the `chunk_size` and `chunk_overlap` parameters, or replace the `RecursiveCharacterTextSplitter` with a more specialized loader (e.g., `MarkdownTextSplitter` for better handling of notes).
    

### 3.4. Changing Retrieval Strategy

To modify the core RAG logic (e.g., to implement advanced hypothetical question generation or better re-ranking):

1. **Locate:** The `query` method in **`rag_agentic.py`**.
    
2. **Modify:** The logic after the initial vector search, focusing on how the retrieved chunks are filtered and combined into the final `raw_context_text` sent to the LLM.