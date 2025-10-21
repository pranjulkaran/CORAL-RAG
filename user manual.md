
# **Agentic RAG - User Guide**

## **Overview**

Agentic RAG is a **Retrieval-Augmented Generation (RAG) system** that allows you to query a local knowledge base for context-grounded answers using **LLM-powered generation**. It supports:

- Indexed document retrieval from PDFs, text, and images
    
- Hybrid LLM + vector database interaction
    
- Local RAG mode (grounded answers) or general chat mode (LLM-only)
    
- Adjustable retrieval parameters (Top K and Top N chunks)
    

This guide helps end-users install dependencies, run the system, and interact with it effectively.

---

## **1. System Requirements**

|Component|Minimum Requirement|
|---|---|
|Python|3.11+|
|Virtual Environment|Recommended (`venv` or `conda`)|
|Ollama LLM|`llama3.2:latest`|
|Vector Database|ChromaDB|
|OCR Engine|Tesseract OCR|
|PDF Parsing|Poppler|
|OS|Windows, Linux, macOS|

---

## **2. Installation**

### **2.1 Clone Repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **2.2 System Dependencies (Windows)**

1. **Poppler** (for PDF parsing)
    
    - Download from the GitHub repo: `[YourRepo]/tools/poppler/`
        
    - Extract and add the `bin` folder to your **system PATH**.
        
2. **Tesseract OCR** (for scanned PDFs/images)
    
    - Download from the GitHub repo: `[YourRepo]/tools/tesseract/`
        
    - Install or extract, then add the path to the executable to your **system PATH**.
        

**Verify Installation**

```bash
tesseract --version
pdftotext -v
```

---

### **2.3 LLM Setup**

- Ensure **Ollama** is installed and running:
    

```bash
ollama list
```

- Required LLM model: `llama3.2:latest`
    

---

## **3. Modes of Operation**

### **3.1 Indexing Documents**

Prepares documents for vector search.

```bash
python main.py --mode index --folder path/to/documents
```

**Notes:**

- Supports PDFs, text files, and images.
    
- Uses Poppler for PDF text extraction.
    
- Uses Tesseract OCR for scanned PDFs or images.
    

---

### **3.2 Query Mode**

Retrieve answers from the RAG database via CLI.

```bash
python main.py --mode query --query "Your question here"
```

**Output includes:**

- Answer from the LLM
    
- Sources used (document filenames)
    
- Context chunks retrieved from the database
    

---

### **3.3 Wipe Database**

Permanently delete all indexed documents.

```bash
python main.py --mode wipe
```

- Requires confirmation (`yes`) before deletion.
    

---

### **3.4 Web Chat Interface**

Launch a Streamlit-based chat UI with RAG and general chat modes:

```bash
python main.py --mode app
```

**Features:**

- Switch between **Agentic RAG mode** and **Regular Chat mode**
    
- View context chunks and sources for RAG responses
    
- Adjustable **Top K / Top N** retrieval settings
    
- Chat history persistence across sessions
    

---

## **4. RAG Interaction Details**

### **4.1 Modes**

- **Agentic RAG (default)**: Retrieves and ranks documents from the database to answer your query.
    
- **Regular Chat**: Uses LLM alone without retrieval.
    

**Mode Switching:**

- `/rag` → Switch to RAG Mode
    
- `/chat` → Switch to LLM-only mode
    

---

### **4.2 Retrieval Tuning**

- **Top K (Retrieve):** Number of candidate chunks fetched from DB
    
- **Top N (Context):** Number of top-ranked chunks sent to the LLM
    

Adjust via Streamlit sidebar or CLI (internal defaults: K=15, N=5).

---

### **4.3 Supported Document Types**

- PDF (`.pdf`) – via Poppler and Tesseract for scanned PDFs
    
- Text (`.txt`)
    
- Images (`.png`, `.jpg`) – OCR applied via Tesseract
    

---

### **4.4 Chat Features**

- Persistent chat history stored locally (`chat_persistence.json`)
    
- Chat bubbles styled in dark mode (customizable in `app.py`)
    
- Expandable sections for viewing context chunks and sources
    

---

## **5. File Organization**

```
/project-root
│
├─ /tools
│   ├─ /poppler/        # Poppler binaries
│   └─ /tesseract/      # Tesseract installer
│
├─ /docs               # Sample documents
├─ ingest_pipeline.py  # Document parsing and embedding
├─ rag_agentic.py      # RAG agent implementation
├─ vector_db_factory.py# ChromaDB connection factory
├─ app.py              # Streamlit interface
├─ main.py             # CLI interface
└─ requirements.txt
```

---

## **6. Troubleshooting**

1. **Ollama LLM not found:**
    
    - Ensure Ollama is running: `ollama list`
        
    - Verify the required model is installed: `llama3.2:latest`
        
2. **Poppler/Tesseract not recognized:**
    
    - Confirm paths are added to **system PATH**
        
    - Open a new terminal and run `tesseract --version` and `pdftotext -v`
        
3. **Database errors:**
    
    - Check ChromaDB configuration
        
    - Use `--mode wipe` cautiously to reset DB
        
4. **RAG answers missing context:**
    
    - Increase Top K/N in the Streamlit sidebar
        
    - Ensure documents are indexed properly
        

---

## **7. Tips**

- Use clear, descriptive queries for best RAG results.
    
- Regularly index new or updated documents for up-to-date answers.
    
- For scanned PDFs, ensure resolution is sufficient for Tesseract OCR.
    

---

**End of User Guide**
