import os
from dotenv import load_dotenv

load_dotenv()

# Path to the master folder containing the source documents (.pdf, .md)
# Used by ingestion_pipeline.py
MASTER_DOCS_PATH = os.getenv("MASTER_DOCS_PATH", r"D:\obsidian notes\Note")

# Path where ChromaDB will store its persistent files
# Used by vector_db_factory.py
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"D:\rag_storage")

# The vector database type to use (currently only 'chroma' is supported)
# Used by vector_db_factory.py
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
