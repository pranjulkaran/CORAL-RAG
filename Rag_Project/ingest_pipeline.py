import os
import hashlib
import asyncio
import time
from datetime import datetime
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from rag_embedder import OllamaBatchEmbedder
from vector_db_factory import get_vector_db
import rich

console = rich.get_console()
from rich.progress import track


class IngestPipeline:
    def __init__(self):
        self.collection = get_vector_db()
        self.embedder = OllamaBatchEmbedder()

    def parse_docs(self, folder):
        docs = []
        files = []
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.endswith((".pdf", ".md")):
                    files.append(os.path.join(root, f))

        for filepath in track(files, description="Parsing documents"):
            try:
                file_mtime = os.path.getmtime(filepath)
                # Query for existing chunks related to this source path
                # FIX: Removed 'ids' from include to resolve the ChromaDB error.
                existing = self.collection.get(where={"source": filepath}, include=["metadatas"])

                if existing["ids"]:  # This check remains valid
                    stored_mtime = existing["metadatas"][0].get("file_mtime", None)

                    if stored_mtime and abs(stored_mtime - file_mtime) < 1:
                        console.print(f"[yellow]Skipping unchanged file:[/yellow] {filepath}")
                        continue
                    else:
                        # File modified: delete old chunks before re-indexing
                        console.print(
                            f"[yellow]File modified. Deleting {len(existing['ids'])} old chunks for:[/yellow] {filepath}")
                        self.collection.delete(ids=existing["ids"])  # IDs are still correctly retrieved here

                loader = PyPDFLoader(filepath) if filepath.endswith(".pdf") else UnstructuredMarkdownLoader(filepath)
                parsed_docs = loader.load()
                # Attach source metadata for later
                for doc in parsed_docs:
                    doc.metadata["source"] = filepath
                docs.extend(parsed_docs)
                console.print(f"[green]Parsed:[/green] {filepath} ({len(parsed_docs)} chunks)")
            except Exception as e:
                # Catch the error and print the full path, as seen in your trace
                console.print(f"[red]Error parsing {filepath}: {e}")
                # CONTINUE processing other files, don't crash the pipeline
        return docs

    async def index_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=60)
        chunks = []
        metadatas = []
        for d in docs:
            text_chunks = splitter.split_text(d.page_content)
            for chunk in text_chunks:
                chunks.append(chunk)
                metadatas.append({
                    "source": d.metadata.get("source", "unknown"),
                    "indexed_at": str(datetime.now()),
                    "file_mtime": os.path.getmtime(d.metadata.get("source", "")) if d.metadata.get("source") else None
                })

        seen = set()
        unique_chunks = []
        unique_metadatas = []
        unique_hashes = []

        for i, chunk in enumerate(chunks):
            h = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_chunks.append(chunk)
                unique_metadatas.append(metadatas[i])
                unique_hashes.append(h[:32])

        embeddings = []
        batch_size = 16
        for i in track(range(0, len(unique_chunks), batch_size), description="Generating embeddings"):
            batch = unique_chunks[i: i + batch_size]
            batch_embeds = await self.embedder.embed_batch(batch)
            embeddings.extend(batch_embeds)

        ids = [str(uuid.UUID(hex_id)) for hex_id in unique_hashes]

        if ids:
            console.print(f"[blue]Sample UUID-formatted ID:[/blue] {ids[0]}")

        if unique_chunks:
            self.collection.add(documents=unique_chunks, embeddings=embeddings, metadatas=unique_metadatas, ids=ids)
            console.print(f"[bold green]Indexed {len(unique_chunks)} unique chunks.")