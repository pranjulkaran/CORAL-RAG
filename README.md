


# CORAL-RAG[Context-Orchestrated Retrieval And Learning RAG]

A modular Retrieval-Augmented Generation (RAG) system with CLI and Streamlit web interface, supporting document indexing, querying, and OCR for PDFs/images. Built for local LLMs with vector database support.

---

## 🌟 Features

- **Indexing**: Parse, embed, and store documents in a vector database.  
- **Querying**: Retrieve and generate answers using top-k re-ranked context chunks.  
- **OCR Support**: Extract text from PDFs and images using Tesseract OCR.  
- **PDF Conversion**: Convert PDFs to images using Poppler for better OCR handling.  
- **Web Interface**: Streamlit-based chat for interactive queries.  
- **Vector DB Management**: Option to wipe the database safely.  
- **Rich Console Output**: Clean CLI formatting with `rich`.  

---

## ⚙️ Requirements

- Python ≥ 3.10
- Virtual environment recommended  

### Python Packages

All required Python packages are listed in `requirements.txt` (install with `pip install -r requirements.txt`). Key dependencies include:

- `chromadb` → Vector database  
- `langchain`, `llama_cpp_python` → LLM integration  
- `sentence-transformers` → Embeddings  
- `PyMuPDF`, `pdf2image`, `PyPDF2` → PDF handling  
- `pillow`, `opencv-python` → Image processing  
- `unstructured.pytesseract` → OCR pipeline  
- `streamlit` → Web UI  
- `rich` → Console formatting  

### System Dependencies

- **Tesseract OCR** (Windows binaries provided in repo)  
- **Poppler utilities** (Windows binaries provided in repo)  

> Ensure both Tesseract and Poppler binaries are in your system PATH or set via environment variables.

---

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/pranjulkaran/CORAL-RAG.git
cd Rag_Project
````

2. Create a virtual environment:
    

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

3. Install Python dependencies:
    

```bash
pip install -r requirements.txt
```

4. Add Tesseract and Poppler to your PATH (or use provided binaries folder).
    

---

## 🏃 Usage

### 1. CLI Modes

- **Index Documents**:
    

```bash
python main.py --mode index --folder "docs"
```

- **Query the Vector Database**:
    

```bash
python main.py --mode query --query "What is Artificial Intelligence?"
```

- **Wipe the Vector Database**:
    

```bash
python main.py --mode wipe
```

- **Launch Streamlit Web App**:
    

```bash
python main.py --mode app
```

---

### 2. Streamlit Web Interface

1. Run:
    

```bash
python main.py --mode app
```

2. Open the URL shown in your terminal (usually `http://localhost:8501`)
    
3. Ask questions interactively, and view source documents and context chunks.
    

---

## 🗂 Folder Structure

```
RAG_Project/
├─ main.py                 # CLI entry point
├─ app.py                  # Streamlit web app
├─ ingest_pipeline.py      # Document parsing & indexing
├─ rag_agentic.py          # RAG query agent
├─ vector_db_factory.py    # Vector database initialization
├─ requirements.txt
└─ README.md
```

---

## ⚡ Notes

- Always activate your virtual environment before running any commands.
    
- For PDF OCR, Poppler is required to convert pages into images.
    
- Ensure your Tesseract installation matches your platform (Windows, Linux, macOS).
    

---

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
    
- [Chromadb Documentation](https://docs.trychroma.com/)
    
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
    
- [Poppler](https://poppler.freedesktop.org/)
    





