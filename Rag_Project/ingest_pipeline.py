import os
import hashlib
import asyncio
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from rag_embedder import OllamaBatchEmbedder
from vector_db_factory import get_vector_db
import rich

console = rich.get_console()

class IngestPipeline:
    def __init__(self):
        self.collection = get_vector_db()
        self.embedder = OllamaBatchEmbedder()

    def parse_docs(self, folder):
        docs = []
        for root, dirs, files in os.walk(folder):
            # Skip hidden .obsidian folder
            dirs[:] = [d for d in dirs if d != ".obsidian"]

            for f in files:
                if not f.endswith((".pdf", ".md")):
                    continue

                path = os.path.join(root, f)

                # Load based on extension
                loader = PyPDFLoader(path) if f.endswith(".pdf") else UnstructuredMarkdownLoader(path)
                try:
                    docs.extend(loader.load())
                except Exception as e:
                    console.print(f"[red]Error parsing {path}: {e}")
        return docs

    async def index_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=60)
        chunks = [d.page_content for d in splitter.split_documents(docs)]
        # Deduplicate
        seen = set()
        filtered = []
        for chunk in chunks:
            h = hashlib.md5(chunk.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                filtered.append(chunk)
        embeddings = await self.embedder.embed_batch(filtered)
        ids = [hashlib.md5(c.encode()).hexdigest() for c in filtered]
        self.collection.add(
            documents=filtered,
            embeddings=embeddings,
            metadatas=[{"indexed_at": str(datetime.now()), "source": "local"}] * len(filtered),
            ids=ids,
        )
        console.print(f"[green]Indexed {len(filtered)} chunks.")
