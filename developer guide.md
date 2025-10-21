
## 🧭 **Developer Guide – Agentic RAG System for Obsidian Notes**

### 📘 Overview

This project is a **modular Agentic RAG (Retrieval-Augmented Generation)** system designed for **Obsidian notes** or any local document set.  
It enables intelligent search, summarization, and reasoning directly over your personal knowledge base.

It combines:

- **Document ingestion + OCR (Tesseract)**
    
- **Text embedding + Vector storage**
    
- **LLM reasoning via the Agentic RAG pipeline**
    
- **Web and CLI interfaces (Streamlit + argparse)**
    
- **Automated cleanup and incremental reindexing**
    

---

## ⚙️ **System Architecture**

```
+----------------------------------------------------------+
|                         main.py                          |
|  - CLI Interface (index / query / wipe / app)            |
+----------------------------------------------------------+
                |               |                |
                ↓               ↓                ↓
       +----------------+  +----------------+  +----------------+
       | ingest_pipeline|  | rag_agentic.py |  | vector_db_factory |
       +----------------+  +----------------+  +----------------+
       | - OCR (tesseract) | - Retrieval, Re-Rank | - DB abstraction |
       | - Parsing         | - Generation (LLM)   | - CRUD ops       |
       | - Chunking        | - Source tracking    |                  |
       | - Hash-based check|                      |                  |
       +----------------+                        +----------------+
                |
                ↓
       +-------------------+
       |    Vector Store    |
       | (Chroma / FAISS)   |
       +-------------------+
                |
                ↓
       +-------------------+
       |     LLM Engine     |
       | (OpenAI / Local)   |
       +-------------------+
                |
                ↓
       +-------------------+
       |  Streamlit Web UI |
       |     (app.py)      |
       +-------------------+
```

---

## 🧩 **Core Components**

### 1️⃣ **main.py**

- **Purpose**: Central CLI entry point for all operations.
    
- **Modes**:
    
    - `--mode index --folder <path>` → Parse + embed new/changed files.
        
    - `--mode query --query "<question>"` → Retrieve and answer.
        
    - `--mode wipe` → Delete all embeddings safely.
        
    - `--mode app` → Launch the Streamlit web UI.
        

**Highlights:**

- Async indexing with `asyncio.run()`.
    
- Rich console output.
    
- Safe wipe confirmation.
    
- Clean subprocess handling for `streamlit run app.py`.
    

---

### 2️⃣ **ingest_pipeline.py**

Handles document ingestion and preprocessing.

**Responsibilities:**

- Parse documents (PDF, Markdown, TXT, Images, etc.)
    
- Use **Tesseract OCR** for scanned or image-based text extraction.
    
- Clean, chunk, and hash files.
    
- Call embedding model for vectorization.
    
- Maintain incremental indexing (index only new/modified docs).
    
- Cleanup deleted files from DB.
    

**Example Flow:**

```python
pipeline = IngestPipeline()
pipeline.cleanup_deleted_files(folder)
docs = pipeline.parse_docs(folder)
await pipeline.index_docs(docs)
```

**Dependencies:**

- `pytesseract`
    
- `pdfplumber` or `PyMuPDF` (for text PDF)
    
- `poltter` (custom preprocessing or visualization utility)
    
- `langchain-text-splitter` or custom chunker
    

---

### 3️⃣ **vector_db_factory.py**

Factory module to create and manage vector database connections.

**Supported backends:**

- `Chroma`
    
- (optional) `FAISS`, `Weaviate`, or others.
    

**Responsibilities:**

- Initialize or connect to existing DB.
    
- Expose methods like `add_documents`, `query`, `delete`, `wipe`.
    

**Example:**

```python
from vector_db_factory import get_vector_db
db = get_vector_db()
results = db.query("What is attention mechanism?")
```

---

### 4️⃣ **rag_agentic.py**

Implements the **Agentic Retrieval-Augmented Generation** logic.

**Core Steps:**

1. **Retrieve** top-k relevant chunks from the vector store.
    
2. **Re-rank** results using semantic similarity or reranker model.
    
3. **Generate** an answer using the LLM.
    
4. **Return** structured result:
    
    ```python
    {
        "answer": "…",
        "sources": ["doc1.md", "doc2.pdf"],
        "context_chunks": ["...", "..."]
    }
    ```
    

**Advanced Features:**

- Support for **multi-turn memory** or chain-of-thought reasoning.
    
- Future: integrate tool-use or external API calls.
    

---

### 5️⃣ **app.py**

A **Streamlit-based chat interface** for end users.

**Features:**

- Chat UI with persistent memory.
    
- Real-time retrieval from indexed notes.
    
- Optional visualization using **poltter** (for embeddings, document maps, etc.).
    
- Supports both text and OCR-based inputs.
    

Run via:

```bash
python main.py --mode app
```

---

## 🧰 **Integrations**

### 🔤 **Tesseract OCR**

Used for extracting text from scanned PDFs or image-based notes.

**Install:**

```bash
sudo apt install tesseract-ocr
pip install pytesseract
```

**Usage in pipeline:**

```python
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open("scan.jpg"))
```

---

### 📊 **Poltter**

A custom or auxiliary visualization / preprocessing tool (context-dependent).  
If you’re using it to visualize embedding spaces or note relationships:

**Usage Examples:**

```python
from poltter import EmbeddingVisualizer
viz = EmbeddingVisualizer()
viz.plot_embeddings(embeddings, labels=doc_names)
```

---

## 🧪 **Developer Setup**

### 🧱 **Installation**

```bash
git clone <repo_url>
cd rag-system
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 🧾 **Dependencies (requirements.txt)**

```
rich
streamlit
chromadb
pdfplumber
pytesseract
Pillow
openai
tiktoken
poltter
numpy
langchain
asyncio
```

---

## 🧠 **Typical Development Flow**

### 🔹 Indexing your Obsidian Notes

```bash
python main.py --mode index --folder "C:\Users\<you>\ObsidianVault"
```

### 🔹 Querying

```bash
python main.py --mode query --query "What are my notes about trading psychology?"
```

### 🔹 Wiping all data

```bash
python main.py --mode wipe
```

### 🔹 Launching the Web UI

```bash
python main.py --mode app
```

---

## 🧩 **Project Extension Ideas**

- Add **cross-document graph search** (vector + link-based retrieval).
    
- Add **summarization agent** that builds context maps.
    
- Integrate **local LLM (e.g., Ollama, LM Studio)** for offline queries.
    
- Add **metadata filters** (date, tags, topic).
    
- Real-time **Obsidian vault watcher** to auto reindex on file changes.
    

---

## ⚡ **Troubleshooting**

|Issue|Cause|Fix|
|---|---|---|
|`streamlit: command not found`|Streamlit not installed in venv|`pip install streamlit`|
|`TesseractNotFoundError`|Tesseract not on PATH|Add Tesseract to PATH or specify `pytesseract.pytesseract.tesseract_cmd`|
|`DB locked / schema mismatch`|Interrupted indexing|Delete `.chromadb` folder and re-index|
|Slow indexing|OCR or LLM embedding latency|Batch embeddings / enable async parallelism|

---

## 🧑‍💻 **Developer Notes**

- Use `asyncio` for all embedding-heavy workloads.
    
- Log internal RAG flow with `rich` or `logging`.
    
- Keep `app.py` lightweight — heavy logic should remain in backend modules.
    
- Version your vector DB schema when changing embedding models.
    
