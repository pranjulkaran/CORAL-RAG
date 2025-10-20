# 🧠 Personal RAG Knowledge Base

This project implements a highly efficient and robust Retrieval-Augmented Generation (RAG) pipeline designed to query diverse personal documents, including structured notes (`.md`) and rich external resources (`.pdf`). It leverages cutting-edge embedding and local LLM technology via Ollama to ensure private, accurate, and context-aware responses.

## 🚀 Key Features

- **State-of-the-Art (SOTA) Retrieval:** Uses the **`mxbai-embed-large`** model for embedding, which is recognized for providing superior semantic search accuracy, ensuring the AI retrieves the most relevant context from your top-tier PDFs and notes.
    
- **High-Speed Indexing:** Implements batched processing (1000 chunks per API call) for massively reduced indexing time.
    
- **Intelligent Incremental Updates:** Documents are checked via **Mtime** and **Content Hash**. If a document is merely moved or renamed, the system performs a fast metadata **UPSERT** instead of an expensive, full re-embedding.
    
- **Safe Multi-Folder Management:** Uses **Scoped Cleanup** to safely index documents from separate folders (`--folder "PathA"`, `--folder "PathB"`) without accidentally deleting data from other indexed paths.
    

## 🛠️ Usage and Setup

### 1. Indexing Documents

Use the `main.py` script with the `--mode index` flag to process a specific folder recursively.

```
# Example: Indexing a source folder
python main.py --mode index --folder "D:\obsidian notes\Note\Source materials\Business analysis"
```

### 2. Launching the Chat Interface

Start the Streamlit application to begin chatting with your indexed knowledge base.

```
streamlit run app.py
```

## 🔗 Project Architecture Backlinks

|   |   |   |
|---|---|---|
|**File**|**Description**|**Focus**|
|[**`ingestion_pipeline.py`**](https://www.google.com/search?q=ingestion_pipeline.py "null")|The core logic for file parsing, chunking, and the **advanced efficiency mechanisms** (Hash/Mtime check, Move Detection, Scoped Cleanup).|Efficiency & Stability|
|[**`rag_embedder.py`**](https://www.google.com/search?q=rag_embedder.py "null")|Handles the asynchronous, batched API calls to the local Ollama embedding service (`mxbai-embed-large`).|API Communication|
|[**`vector_db_factory.py`**](https://www.google.com/search?q=vector_db_factory.py "null")|Initialization and connection logic for the ChromaDB vector store.|Persistence Layer|
|[**`main.py`**](https://www.google.com/search?q=main.py "null")|Entry point for command-line operations (index mode).|Application Entry|