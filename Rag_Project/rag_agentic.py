import os
import re
import chromadb
import ollama
import asyncio
# Assuming these imports are available in the project environment
from vector_db_factory import get_vector_db
from rag_embedder import OllamaBatchEmbedder


class AgenticRAG:
    """
    Implements the Retrieval-Augmented Generation (RAG) agent using a two-stage
    retrieval process (Candidate Generation + Simulated Re-Ranking) and Ollama
    for both embeddings and generation.
    """

    def __init__(self):
        # Initialize the database connection
        self.collection = get_vector_db()

        # Initialize the Ollama embedder for generating query vectors (1024-dim)
        # We need to create a new instance here to avoid circular dependency issues
        # if the embedder was imported globally in this structure.
        self.embedder = OllamaBatchEmbedder()

        # LLM to be used for final answer generation
        self.model = "llama3.2:latest"

        # Configuration for Two-Stage Retrieval
        # Stage 1: Initial number of candidates retrieved from Vector DB
        self.top_k_retrieve = 15
        # Stage 2: Final number of best chunks passed to the LLM (Re-Ranked subset)
        self.top_n_rank = 5

    def _rerank(self, query: str, results: dict):
        """
        Simulates the re-ranking step. Since we don't use a dedicated cross-encoder
        here, we rely on Chroma's similarity score ordering and simply truncate
        the results to the top_n_rank.
        """

        # ChromaDB query results are lists nested inside another list (e.g., [[]])
        # We assume the results are already sorted by distance (similarity)
        documents = results["documents"][0]
        metadata = results["metadatas"][0]
        distances = results["distances"][0]

        # Select the final, most relevant subset (top N)
        top_documents = documents[:self.top_n_rank]
        top_metadata = metadata[:self.top_n_rank]
        top_distances = distances[:self.top_n_rank]

        return top_documents, top_metadata, top_distances

    async def retrieve(self, query: str):
        """
        Retrieves relevant context chunks from the vector database.
        This is an async method because it calls the asynchronous embedder.
        """

        # 1. Generate the 1024-dimension query vector using the Ollama embedder
        # Note: embed_batch is used for consistency, even with a single query
        query_embedding_list = await self.embedder.embed_batch([query])
        query_embedding = query_embedding_list[0]

        # 2. Query the vector store for a large number of candidate chunks (Stage 1)
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use the Ollama vector
            n_results=self.top_k_retrieve,  # Retrieve the larger candidate set
            include=['documents', 'metadatas', 'distances']
        )

        if not results.get("documents") or not results["documents"][0]:
            # No results found
            return "", [], []

        # 3. Apply Re-Ranking/Filtering to get the best N chunks (Stage 2)
        documents, metadata, _ = self._rerank(query, results)

        # --- Whitespace Normalization (ENHANCED LOGIC) ---
        normalized_documents = []
        for doc in documents:
            normalized_doc = doc

            # 1. Fix CamelCase/Run-on Words
            normalized_doc = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized_doc)

            # 2. Fix Punctuation Run-ons
            normalized_doc = re.sub(r'([\.?!,:;])([a-zA-Z0-9])', r'\1 \2', normalized_doc)

            # 3. Fix Digit Run-ons (Letter followed by digit)
            normalized_doc = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', normalized_doc)

            # 4. Fix Digit Run-ons (Digit followed by letter)
            normalized_doc = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', normalized_doc)

            normalized_documents.append(normalized_doc)

        # Create context string from re-ranked and normalized documents
        context = "\n\n---\n\n".join(normalized_documents)

        # 4. Return the context string, metadata, and the list of normalized documents
        return context, metadata, normalized_documents

    def generate(self, query: str, context: str, chat_history: list):
        """
        Generates the final answer using the LLM based on the retrieved context.
        """
        # Format the conversational history for the LLM prompt
        history_str = "\n".join([f"{h['speaker']}: {h['message']}" for h in chat_history])

        # FINALIZED SYSTEM PROMPT: Strong directive for grounded generation
        system_prompt = (
            "You are a helpful assistant. Use the provided CONTEXT to formulate your answer. "
            "If the context contains information that directly or indirectly answers the question, summarize and state it clearly. "
            "**Do not state that the information is 'inferred' or 'not explicitly mentioned' if the components are present in the text.** "
            "If the context does not contain relevant information, state that you cannot answer based on the provided documents. "
            "Always include source citations at the end of your answer."
        )

        # User message combining history, context, and the new query
        user_message_content = (
            f"HISTORICAL CONVERSATION:\n{history_str}\n\n"
            f"NEW CONTEXT (Use this for your answer, this context has been pre-filtered for relevance):\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Please answer the QUESTION based ONLY on the provided CONTEXT."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            # Options for detailed and long response
            options={"temperature": 0.7, "num_ctx": 8000, "num_predict": 4000}
        )

        return response["message"]["content"]

    def query(self, question: str, chat_history: list = None):
        """
        The main synchronous entry point for the query process.
        """
        chat_history = chat_history if chat_history is not None else []

        # Use asyncio.run to execute the async retrieval function
        try:
            # We call the async method from the sync context
            context, metadata, documents = asyncio.run(self.retrieve(question))
        except Exception as e:
            # Handle retrieval errors gracefully
            print(f"Error during async retrieval: {e}")
            return {"answer": f"An unexpected error occurred during context retrieval: {e}", "sources": [],
                    "context_chunks": []}

        if not context:
            return {"answer": "I could not find any relevant documents in the database to answer your question.",
                    "sources": [], "context_chunks": []}

        # Generate the final answer
        answer = self.generate(question, context, chat_history)

        # Extract unique source names (file paths)
        unique_sources = list(set(md.get("source", "Unknown Source") for md in metadata))

        # The documents list here only contains the final, re-ranked and normalized chunks
        return {"answer": answer, "sources": unique_sources, "context_chunks": documents}
