# 🧱 System Architecture: Agentic RAG Pipeline

This document outlines the architecture and data flow of the Retrieval-Augmented Generation (RAG) system, emphasizing the three core layers: **Ingestion**, **Storage/Retrieval**, and **Generation**.

## 1. Conceptual Flow (The Two-Stage Agentic Pipeline)

The system is designed around two main phases—Ingestion (for building the knowledge base) and Query (for generating answers). The Query phase uses a multi-step "Agentic" approach for higher quality results.

|Phase|Steps|Key Components|
|---|---|---|
|**Ingestion** (Offline)|**Load & Process:** Load documents, calculate **Content Hash** and **Mtime**. **Chunk:** Split documents into small, indexed segments. **Embed:** Convert chunks to vectors using `mxbai-embed-large`. **Index:** Store vectors in ChromaDB.|`ingestion_pipeline.py`, `rag_embedder.py`, ChromaDB|
|**Query** (Online)|**Retrieve (k):** Embed user query and fetch `top_k_retrieve` candidates from the Vector Store. **Re-Rank (n):** An internal agent filters and re-ranks the candidates based on relevance to the full context. **Generate:** Send the top `top_n_rank` chunks + the user query to the LLM.|`rag_agentic.py`, Ollama LLM (`llama3.2:latest`)|

## 2. Core Components and Technology Stack

The pipeline is built on an open-source, local-first stack, ensuring privacy and full control over model selection.

### 2.1. The Knowledge Index (ChromaDB)

- **Role:** Acts as the persistent storage layer for the vector embeddings and metadata.
    
- **Key Feature:** Supports **UPSERT** functionality, which is critical for the pipeline's efficiency, allowing for fast metadata updates (like changing a file's path) without re-embedding.
    
- **Implementation:** Managed by **`vector_db_factory.py`** and stored locally at the path defined in `config.py`.
    

### 2.2. The Vectorizer (Ollama + `mxbai-embed-large`)

- **Role:** Responsible for converting both document chunks (Ingestion) and the user's query (Query) into dense numerical vectors.
    
- **Key Feature:** Uses the **`mxbai-embed-large`** model, which provides State-of-the-Art (SOTA) semantic accuracy for its size, ensuring precise retrieval.
    
- **Implementation:** Handled by **`rag_embedder.py`** using asynchronous, batched calls to the local Ollama server.
    

### 2.3. The Ingestion Engine (Optimized for Efficiency)

- **Role:** Manages the entire data preparation workflow, focusing heavily on incremental updates and stability.
    
- **Key Feature:** Implements advanced logic for **Scoped Cleanup** (protects indexed data from other folders) and **Content Hash** comparison (prevents costly re-embedding of moved files).
    
- **Implementation:** Contained entirely within **`ingestion_pipeline.py`**.
    

### 2.4. The Generator and Agent (LLM + Logic)

- **Role:** Orchestrates the multi-stage retrieval process and ultimately synthesizes the final answer.
    
- **Key Feature:** Uses a two-step retrieval process (Retrieve `k` then Re-Rank to `n`) to reduce noise and provide the LLM with the highest quality context.
    
- **Implementation:** Contained within **`rag_agentic.py`** (implied) and uses the locally running `llama3.2:latest` model via Ollama.