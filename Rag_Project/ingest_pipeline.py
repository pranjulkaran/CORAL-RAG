import os
import hashlib
import asyncio
from datetime import datetime
import uuid
import rich
import json  # CRITICAL: Needed for metadata sanitization
from rich.progress import track
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Changed to use UnstructuredPDFLoader for better handling of complex PDF layouts
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredMarkdownLoader

# External dependencies (assumed to be available in the project structure)
from rag_embedder import OllamaBatchEmbedder
from vector_db_factory import get_vector_db

# --- Configuration & Memory Management Constants ---
# NOTE: The constants are now assumed to be in the global scope or imported from config.py
EMBEDDING_API_BATCH_SIZE = 1000  # Default to high speed
VECTOR_DB_COMMIT_BATCH_SIZE = 1000

console = rich.get_console()


def _get_file_sha256(filepath, block_size=65536):
    """Calculates SHA256 hash of a file efficiently without loading it all into memory."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        console.print(f"[red]Error reading file for hash {filepath}: {e}[/red]")
        return None


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
        Scans a folder, checks files for modifications using mtime and content hash.
        CRITICAL UPDATE: Now checks for identical content (file_hash) at a new location
        to prevent unnecessary re-embedding of moved files.
        """
        docs = []
        files = []

        # 1. Collect all relevant files (.pdf, .md)
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.endswith((".pdf", ".md")):
                    files.append(os.path.join(root, f))

        # 2. Get all unique source paths and their associated hashes from the database
        db_results = self.collection.get(include=['metadatas'])

        # Mapping: file_hash -> set of (source_path, mtime)
        hash_to_sources = {}
        for md in db_results.get('metadatas', []):
            f_hash = md.get('file_hash')
            f_source = md.get('source')
            f_mtime = md.get('file_mtime')
            if f_hash and f_source:
                if f_hash not in hash_to_sources:
                    hash_to_sources[f_hash] = set()
                # We store the mtime from the database to ensure we use the same mtime consistently
                hash_to_sources[f_hash].add((f_source, f_mtime))

        # 3. Processing Loop
        for filepath in track(files, description="Parsing documents"):
            try:
                current_source_key = os.path.abspath(filepath)
                file_mtime = os.path.getmtime(filepath)
                file_hash = _get_file_sha256(filepath)

                if not file_hash:
                    continue  # Skip if hashing failed

                # Check 1: Does the content hash already exist in the database?
                if file_hash in hash_to_sources:

                    found_at_current_location = False

                    # Check if the content is already indexed at the CURRENT location
                    for source, stored_mtime in hash_to_sources[file_hash]:
                        if source == current_source_key:
                            found_at_current_location = True

                            # Check 1a: Content exists and is at the current location. Check mtime.
                            if stored_mtime and abs(stored_mtime - file_mtime) < 1:
                                console.print(
                                    f"[yellow]Skipping unchanged file (mtime and hash match):[/yellow] {os.path.basename(filepath)}")
                                break  # Skip to the next file

                    if found_at_current_location:
                        # Continue to next file if the file is unchanged
                        continue

                    # Check 1b: Content exists, but is NOT at the current location. (MOVED FILE)
                    if not found_at_current_location:
                        console.print(f"[cyan]Content match found. File MOVED:[/cyan] {os.path.basename(filepath)}")

                        # Delete the old source chunks associated with this hash
                        for old_source, _ in hash_to_sources[file_hash]:
                            if old_source != current_source_key:
                                console.print(
                                    f"[yellow]Deleting old chunks for moved file:[/yellow] {os.path.basename(old_source)}")
                                self.collection.delete(where={"source": old_source})

                        # Proceed to re-index below. Since the content is the same,
                        # the chunk IDs will perform a fast UPSERT to update the metadata.

                # Check 2: File is new or modified (mtime or hash check failed). Proceed with loading.

                # Check for existing chunks related to this current path
                existing_chunks_at_current_path = self.collection.get(
                    where={"source": current_source_key},
                    include=["metadatas"]
                )

                if existing_chunks_at_current_path["ids"]:
                    # If we are here, it means the mtime/hash comparison failed, so it must be re-indexed.
                    console.print(
                        f"[yellow]File modified. Deleting {len(existing_chunks_at_current_path['ids'])} old chunks for:[/yellow] {os.path.basename(filepath)}")
                    self.collection.delete(ids=existing_chunks_at_current_path["ids"])

                # Load the document
                # --- Switched to UnstructuredPDFLoader ---
                loader = UnstructuredPDFLoader(filepath) if filepath.endswith(".pdf") else UnstructuredMarkdownLoader(
                    filepath)

                # --- CRITICAL LOAD ERROR HANDLING ADDED ---
                try:
                    parsed_docs = loader.load()
                except Exception as load_e:
                    # Catch and log any deep errors during the parsing process and continue
                    console.print(
                        f"[bold red]CRITICAL LOAD ERROR:[/bold red] Failed to parse {os.path.basename(filepath)}. Skipping. Reason: {load_e}")
                    continue
                    # -----------------------------------------------

                # Attach source metadata for later
                for doc in parsed_docs:
                    doc.metadata["source"] = current_source_key
                    doc.metadata["file_mtime"] = file_mtime
                    doc.metadata["file_hash"] = file_hash  # Store the hash with every chunk

                docs.extend(parsed_docs)
                console.print(
                    f"[green]Parsed:[/green] {os.path.basename(filepath)} ({len(parsed_docs)} pages/sections)")

            except Exception as e:
                console.print(f"[red]Error processing file {os.path.basename(filepath)} (pre-load issue):[/red] {e}")

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
        chunks_map = {}

        for d in docs:
            # --- RESILIENCE CHECK ---
            if not d.page_content or not isinstance(d.page_content, str) or d.page_content.strip() == "":
                source_file = d.metadata.get('source', 'Unknown File')
                console.print(
                    f"[bold red]Skipping Document:[/bold red] '{os.path.basename(source_file)}'. Document content is empty or invalid.")
                continue

            # Use 'ignore' error handling for encoding when hashing
            text_chunks = splitter.split_text(d.page_content)

            if not text_chunks:
                source_file = d.metadata.get('source', 'Unknown File')
                console.print(
                    f"[bold yellow]Skipping Document:[/bold yellow] '{os.path.basename(source_file)}'. No chunks generated (content too short or sparse).")
                continue

            for chunk in text_chunks:
                h = hashlib.sha256(chunk.encode('utf-8', errors='ignore')).hexdigest()
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, h))

                if chunk_id in chunks_map:
                    continue

                metadata = d.metadata.copy()
                metadata["indexed_at"] = str(datetime.now())

                # --- Metadata Sanitization (CRITICAL FOR CHROMA DB VALIDATION) ---
                sanitized_metadata = {}
                for key, value in metadata.items():
                    # ChromaDB only supports str, int, float for metadata values
                    if isinstance(value, (str, int, float)):
                        # Handle potential NaN, Inf/-Inf floats which break JSON/Chroma
                        if isinstance(value, float) and (value != value or value in [float('inf'), float('-inf')]):
                            sanitized_metadata[key] = str(value)
                        else:
                            sanitized_metadata[key] = value
                    else:
                        # Convert all other complex types (e.g., lists, dicts) to JSON string
                        try:
                            sanitized_metadata[key] = json.dumps(value)
                        except:
                            # Fallback to string representation if JSON serialization fails
                            sanitized_metadata[key] = str(value)

                # ------------------------------------------------------------------

                chunks_map[chunk_id] = {
                    "chunk": chunk,
                    "id": chunk_id,
                    "metadata": sanitized_metadata  # Use sanitized metadata
                }

        all_chunks_data = list(chunks_map.values())

        num_chunks_to_add = len(all_chunks_data)
        if num_chunks_to_add == 0:
            console.print("[bold yellow]No new unique chunks found to embed or index.[/bold yellow]")
            return

        console.print(f"\n[bold blue]Total unique chunks to process (will use UPSERT):[/bold blue] {num_chunks_to_add}")
        console.print(f"--- Starting Batched Embedding and Indexing ---")

        # 2. Batched Embedding and Indexing
        for i in track(range(0, num_chunks_to_add, EMBEDDING_API_BATCH_SIZE), description="Generating and Indexing"):

            embed_start = i
            embed_end = min(i + EMBEDDING_API_BATCH_SIZE, num_chunks_to_add)

            batch_data = all_chunks_data[embed_start:embed_end]
            batch_chunks = [d['chunk'] for d in batch_data]

            # Generate embeddings asynchronously
            batch_embeddings = await self.embedder.embed_batch(batch_chunks)

            # Add embeddings to the batch_data objects for later indexing
            for j, embed in enumerate(batch_embeddings):
                batch_data[j]["embedding"] = embed

            # --- Indexing Commit Batch ---

            commit_chunks = [d['chunk'] for d in batch_data]
            commit_embeddings = [d['embedding'] for d in batch_data]
            commit_metadatas = [d['metadata'] for d in batch_data]
            commit_ids = [d['id'] for d in batch_data]

            # Since the IDs in commit_ids are now guaranteed to be unique within this batch
            # the ChromaDB add/upsert operation will succeed.
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
        console.print("\n--- Starting Scoped Stale Chunk Cleanup ---")

        # 1. Normalize the path for filtering
        scope_path_abs = os.path.abspath(master_docs_path)

        # 2. Get all unique source paths currently in the database
        db_results = self.collection.get(include=['metadatas'])
        sources_in_db = set(md.get('source') for md in db_results.get('metadatas', []) if md.get('source'))

        # 3. Filter DB sources to only include those within the current scope
        sources_in_scope = set()
        for source in sources_in_db:
            if source.startswith(scope_path_abs):
                sources_in_scope.add(source)

        if not sources_in_scope:
            console.print("No previously indexed files found within the current scope for cleanup.")
            return

        # 4. Get all files currently on disk within the specified master_docs_path (normalized)
        files_on_disk_in_scope = set()
        for root, _, fs in os.walk(master_docs_path):
            for f in fs:
                if f.endswith((".pdf", ".md")):
                    files_on_disk_in_scope.add(os.path.abspath(os.path.join(root, f)))

        # 5. Determine stale sources: these are files that are in the scope but NOT on disk
        stale_sources = sources_in_scope - files_on_disk_in_scope

        if stale_sources:
            paths_to_delete = stale_sources

            console.print(
                f"[yellow]Found {len(paths_to_delete)} source files deleted from the scope on disk that need cleanup.[/yellow]")
            for source in paths_to_delete:
                console.print(f"    Removing stale chunks for: {os.path.basename(source)}")
                try:
                    # Delete by metadata filter
                    self.collection.delete(where={"source": source})
                    console.print(f"    [green]SUCCESS:[/green] Removed chunks for {os.path.basename(source)}")
                except Exception as e:
                    console.print(f"    [red]ERROR:[/red] Failed to remove stale chunks for {source}. Reason: {e}")
        else:
            console.print("No stale source files found in the vector database.")
