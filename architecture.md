
# 🏗️ Agentic RAG System – Architecture

## 1️⃣ **Overview**

The Agentic RAG System is a **Retrieval-Augmented Generation (RAG) engine** designed to leverage **local documents** (like PDFs, Markdown, or images) and provide **context-aware responses** via an LLM.

It integrates:

- **Vector database for embeddings**
    
- **Tesseract OCR** for scanned documents
    
- **Poppler** for PDF text extraction
    
- **Streamlit Web UI** for interactive chat
    

---

## 2️⃣ **High-Level Architecture**

```
+------------------------------------------------------------+
|                        main.py                             |
|  CLI Entry Point: index / query / wipe / app               |
+------------------------------------------------------------+
        |               |                |
        ↓               ↓                ↓
+----------------+  +----------------+  +----------------+
| ingest_pipeline|  | rag_agentic.py |  | vector_db_factory|
+----------------+  +----------------+  +----------------+
| - File parsing |  | - Retrieval    |  | - DB connection |
| - PDF parsing  |  | - Re-rank      |  | - CRUD ops      |
|   (Poppler)    |  | - LLM generation|  | - Wipe support  |
| - OCR (Tesseract)| | - Source tracking|                  |
| - Text chunking|  +----------------+  +----------------+
| - Hash checks  |        
| - Incremental  |        
+----------------+        
        |
        ↓
+---------------------+
|   Vector Database    |
|   (Chroma / FAISS)   |
+---------------------+
        |
        ↓
+---------------------+
|     LLM Engine       |
| (Ollama / OpenAI)    |
+---------------------+
        |
        ↓
+---------------------+
|  Streamlit Web UI    |
|       (app.py)       |
+---------------------+
```

---

## 3️⃣ **Module Responsibilities**

### **main.py**

- CLI entry point for all operations.
    
- Modes:
    
    - `index` → Parse + embed new/changed files
        
    - `query` → Ask RAG for answers
        
    - `wipe` → Delete all vectors
        
    - `app` → Launch Streamlit web interface
        
- Uses `asyncio` for indexing.
    

### **ingest_pipeline.py**

- Handles document ingestion:
    
    - File parsing (PDF, Markdown, TXT)
        
    - **Poppler** for PDF text extraction
        
    - **Tesseract OCR** for scanned docs/images
        
    - Text chunking & hashing
        
    - Incremental indexing
        
- Sends processed chunks to **Vector DB**.
    

### **vector_db_factory.py**

- Manages the vector database connection:
    
    - Create / fetch collections
        
    - CRUD operations
        
    - Wipe / cleanup
        
- Backend: **Chroma**, optionally FAISS.
    

### **rag_agentic.py**

- Implements **RAG logic**:
    
    1. Retrieve top-k relevant chunks
        
    2. Re-rank chunks (semantic similarity)
        
    3. Generate answer via LLM
        
    4. Return structured results:
        
        ```json
        {
            "answer": "...",
            "sources": ["doc1.pdf", "doc2.md"],
            "context_chunks": ["...", "..."]
        }
        ```
        

### **app.py**

- Streamlit-based chat interface:
    
    - Interactive Q&A
        
    - Persistent chat history
        
    - Contextual RAG responses
        

---

## 4️⃣ **Data Flow**

```
User Input
   ↓
+-----------------------+
|  main.py CLI / app.py  |
+-----------------------+
   ↓
[If indexing] → ingest_pipeline → Poppler / OCR → Chunk / Hash → Vector DB
[If query]   → rag_agentic → Retrieve & Re-rank → LLM → Answer
   ↓
Web / CLI Output (Answer + Sources + Chunks)
```

---

## 5️⃣ **Key Integrations**

|Tool|Purpose|
|---|---|
|**Tesseract OCR**|Extract text from scanned PDFs/images|
|**Poppler**|Extract text content from PDFs|
|**Chroma / FAISS**|Vector storage for semantic search|
|**Streamlit**|Web-based chat interface|
|**Ollama / OpenAI LLM**|Generates answers based on retrieved context|

---

## 6️⃣ **Developer Notes**

- Async indexing ensures performance on large datasets.
    
- Modular design:
    
    - ingestion → vectorization → retrieval → generation
        
- Web UI is lightweight; heavy processing stays in backend modules.
    
- History & context management for multi-turn conversations.
    
- Supports incremental re-indexing & cleanup of deleted files.
    

---

## 7️⃣ **Future Enhancements**

- Cross-document graph search.
    
- Metadata filtering (tags, dates, topics).
    
- Local LLM fallback for offline usage.
    
- Summarization or document linking agent.
    
- Real-time vault watcher for automatic indexing.
    
