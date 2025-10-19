from config import VECTOR_DB, CHROMA_DB_PATH


def get_vector_db():
    """
    Initializes and returns the vector database collection based on configuration.

    Currently supports ChromaDB, providing persistence and collection management.

    Returns:
        chromadb.Collection: The ChromaDB collection object for RAG documents.
    """
    if VECTOR_DB.lower() == "chroma":
        # Imports are done locally to avoid errors if the chosen DB is not installed
        from chromadb.config import Settings
        import chromadb

        # Initialize a persistent ChromaDB client using the configured path
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))

        # Get or create the main collection where all RAG documents are stored
        return client.get_or_create_collection("rag_docs")
    else:
        # Raise an error if the VECTOR_DB variable is set to an unsupported value
        raise ValueError(f"Unsupported VECTOR_DB specified in config: {VECTOR_DB}")
