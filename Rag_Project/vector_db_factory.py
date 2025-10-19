from config import VECTOR_DB, CHROMA_DB_PATH

def get_vector_db():
    if VECTOR_DB.lower() == "chroma":
        from chromadb.config import Settings
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
        return client.get_or_create_collection("rag_docs")
    else:
        raise ValueError(f"Unsupported VECTOR_DB: {VECTOR_DB}")
