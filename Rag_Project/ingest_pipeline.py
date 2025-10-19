import os
import hashlib
import asyncio
from datetime import datetime
import uuid
import rich
from rich.progress import track
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

# External dependencies (assumed to be available in the project structure)
from rag_embedder import OllamaBatchEmbedder
from vector_db_factory import get_vector_db

console = rich.get_console()

# --- Configuration & Memory Management Constants ---

# Max number of text chunks to send to the embedding model API (e.g., Ollama)
# in a single request. This is crucial for managing VRAM/RAM for large models.
EMBEDDING_API_BATCH_SIZE = 300

# Max number of documents to commit to the vector database at once.
# This is usually higher than the embedding batch size.
VECTOR_DB_COMMIT_BATCH_SIZE = 1000


class IngestPipeline:
    """
    Handles incremental loading, chunking, embedding, and indexing of documents
    into the vector database.
    """

    def __init__(self):
        self.collection = get_vector_db()
        self.embedder = OllamaBatchEmbedder()

    def parse_docs(self, folder):
        """
        Scans a folder, checks files for modifications using mtime,
        loads and prepares documents that are new or updated.
        """
        docs = []
        files = []

        # Collect all relevant files (.pdf, .md)
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.endswith((".pdf", ".md")):
                    files.append(os.path.join(root, f))

        for filepath in track(files, description="Parsing documents"):
            try:
                file_mtime = os.path.getmtime(filepath)

                # Query for existing chunks related to this source path
                # FIX: Use include=["metadatas"] to resolve the ChromaDB error while still
                # getting enough info to check for modifications.
                existing = self.collection.get(where={"source": filepath}, include=["metadatas"])

                # Check if file has existing chunks AND retrieve their IDs for potential deletion
                if existing["ids"]:
                    # Assume mtime is consistently stored across all chunks for a file
                    stored_mtime = existing["metadatas"][0].get("file_mtime", None)

                    # Compare timestamps (allow a small tolerance)
                    if stored_mtime and abs(stored_mtime - file_mtime) < 1:
                        console.print(f"[yellow]Skipping unchanged file:[/yellow] {os.path.basename(filepath)}")
                        continue
                    else:
                        # File modified: delete old chunks before re-indexing
                        console.print(
                            f"[yellow]File modified. Deleting {len(existing['ids'])} old chunks for:[/yellow] {os.path.basename(filepath)}")
                        self.collection.delete(ids=existing["ids"])

                # Load the document
                loader = PyPDFLoader(filepath) if filepath.endswith(".pdf") else UnstructuredMarkdownLoader(filepath)
                parsed_docs = loader.load()

                # Attach source metadata for later
                for doc in parsed_docs:
                    doc.metadata["source"] = filepath
                    doc.metadata["file_mtime"] = file_mtime

                docs.extend(parsed_docs)
                console.print(
                    f"[green]Parsed:[/green] {os.path.basename(filepath)} ({len(parsed_docs)} pages/sections)")

            except Exception as e:
                console.print(f"[red]Error parsing {os.path.basename(filepath)}: {e}")

        return docs

    async def index_docs(self, docs):
        """
        Chunks the documents, queues unique chunks, and processes them in batches
        for embedding and indexing.
        """
        if not docs:
            console.print("[bold yellow]No documents passed for indexing.[/bold yellow]")
            return

        # 1. Chunking and Deduplication
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)

        all_chunks_data = []
        seen_hashes = set()

        for d in docs:
            text_chunks = splitter.split_text(d.page_content)
            for chunk in text_chunks:
                h = hashlib.sha256(chunk.encode('utf-8')).hexdigest()

                if h not in seen_hashes:
                    seen_hashes.add(h)

                    # Store ID as UUIDv5 based on hash for ChromaDB compatibility
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, h))

                    # Generate metadata with required fields
                    metadata = d.metadata.copy()
                    metadata["indexed_at"] = str(datetime.now())
                    # Ensure file_mtime is set for subsequent incremental checks
                    metadata["file_mtime"] = d.metadata.get("file_mtime", None)

                    all_chunks_data.append({
                        "chunk": chunk,
                        "id": chunk_id,
                        "metadata": metadata
                    })

        num_chunks_to_add = len(all_chunks_data)
        if num_chunks_to_add == 0:
            console.print("[bold yellow]No new unique chunks found to embed or index.[/bold yellow]")
            return

        console.print(f"\n[bold blue]Total unique chunks to process:[/bold blue] {num_chunks_to_add}")
        console.print(f"--- Starting Batched Embedding and Indexing ---")

        # 2. Batched Embedding and Indexing

        # Process ALL unique chunks in batches for the embedding API call (smaller limit)
        for i in track(range(0, num_chunks_to_add, EMBEDDING_API_BATCH_SIZE), description="Generating embeddings"):

            # --- Embedding Batch ---
            embed_start = i
            embed_end = min(i + EMBEDDING_API_BATCH_SIZE, num_chunks_to_add)

            batch_data = all_chunks_data[embed_start:embed_end]
            batch_chunks = [d['chunk'] for d in batch_data]

            # Generate embeddings asynchronously
            batch_embeddings = await self.embedder.embed_batch(batch_chunks)

            # Add embeddings to the batch_data objects for later indexing
            for j, embed in enumerate(batch_embeddings):
                batch_data[j]["embedding"] = embed

            # --- Indexing Commit Batch (can be same or different size) ---

            # For simplicity, we commit the same size as the embedding batch here.
            # If your ChromaDB can handle larger commits, you could accumulate
            # multiple embed batches before committing a larger VECTOR_DB_COMMIT_BATCH_SIZE.

            commit_chunks = [d['chunk'] for d in batch_data]
            commit_embeddings = [d['embedding'] for d in batch_data]
            commit_metadatas = [d['metadata'] for d in batch_data]
            commit_ids = [d['id'] for d in batch_data]

            self.collection.add(
                documents=commit_chunks,
                embeddings=commit_embeddings,
                metadatas=commit_metadatas,
                ids=commit_ids
            )

            console.print(
                f"[green]Successfully indexed batch {i // EMBEDDING_API_BATCH_SIZE + 1} ({len(commit_chunks)} chunks).[/green]")

        console.print(f"[bold green]Indexing complete. Total {num_chunks_to_add} unique chunks processed.[/bold green]")

    def cleanup_deleted_files(self, master_docs_path):
        """
        Identifies and removes chunks in the DB whose source file no longer exists on disk.
        """
        console.print("\n--- Starting Stale Chunk Cleanup ---")

        # Get all unique source paths currently in the database
        db_results = self.collection.get(include=['metadatas'])
        sources_in_db = set(md.get('source') for md in db_results.get('metadatas', []) if md.get('source'))

        files_on_disk = set()
        for root, _, fs in os.walk(master_docs_path):
            for f in fs:
                # Ensure only relevant files are checked against sources_in_db
                if f.endswith((".pdf", ".md")):
                    files_on_disk.add(os.path.join(root, f))

        stale_sources = sources_in_db - files_on_disk

        if stale_sources:
            console.print(f"[yellow]Found {len(stale_sources)} source files deleted from disk.[/yellow]")
            for source in stale_sources:
                console.print(f"    Removing stale chunks for: {os.path.basename(source)}")
                try:
                    # Delete by metadata filter
                    self.collection.delete(where={"source": source})
                    console.print(f"    [green]SUCCESS:[/green] Removed chunks for {os.path.basename(source)}")
                except Exception as e:
                    console.print(f"    [red]ERROR:[/red] Failed to remove stale chunks for {source}. Reason: {e}")
        else:
            console.print("No stale source files found in the vector database.")
