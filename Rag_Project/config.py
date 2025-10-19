import os
from dotenv import load_dotenv

load_dotenv()

MASTER_DOCS_PATH = os.getenv("MASTER_DOCS_PATH", r"D:\ragstorage_test")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"D:\rag_storage")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
