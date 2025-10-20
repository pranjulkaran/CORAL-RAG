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
    for both embeddings and generation, now incorporating HyDE (Hypothetical Document Embedding).
    """

    def __init__(self):
        # Initialize the database connection
        self.collection = get_vector_db()

        # Initialize the Ollama embedder for generating query vectors (1024-dim)
        self.embedder = OllamaBatchEmbedder()

        # LLM to be used for final answer generation and HyDE generation
        self.model = "llama3.2:latest"

        # Configuration for Two-Stage Retrieval
        # Stage 1: Initial number of candidates retrieved from Vector DB
        self.top_k_retrieve = 15
        # Stage 2: Final number of best chunks passed to the LLM (Re-Ranked subset)
        self.top_n_rank = 5

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generates a detailed, hypothetical answer using the LLM.
        This hypothetical answer is used to create a better search vector.
        """
        hyde_prompt = (
            "You are an expert researcher. Based on the following user query, "
            "write a detailed, hypothetical answer (about 3-4 sentences). "
            "DO NOT use external knowledge; fabricate a highly plausible response "
            "that captures the semantic complexity of the query. "
            "Do not include any source citations."
        )

        messages = [
            {"role": "system", "content": hyde_prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.3, "num_ctx": 4096, "num_predict": 512}
            )
            return response["message"]["content"]
        except Exception as e:
            # Fallback in case of Ollama error during HyDE generation
            print(f"Error during HyDE generation: {e}. Falling back to original query.")
            return query

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
        Retrieves relevant context chunks from the vector database using HyDE.
        This is an async method because it calls the asynchronous embedder.
        """

        # --- HyDE Step (New) ---
        # 1. Generate the hypothetical document
        hypothetical_document = self._generate_hypothetical_document(query)

        # Determine which text to embed: the HyDE result or the original query if HyDE failed
        search_text = hypothetical_document if hypothetical_document != query else query
        # ------------------------

        # 2. Generate the query vector using the Ollama embedder (using the HyDE document's text)
        query_embedding_list = await self.embedder.embed_batch([search_text])
        query_embedding = query_embedding_list[0]

        # 3. Query the vector store for a large number of candidate chunks (Stage 1)
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Use the HyDE vector
            n_results=self.top_k_retrieve,  # Retrieve the larger candidate set (dynamic K)
            include=['documents', 'metadatas', 'distances']
        )

        if not results.get("documents") or not results["documents"][0]:
            # No results found
            return "", [], []

        # 4. Apply Re-Ranking/Filtering to get the best N chunks (Stage 2)
        documents, metadata, _ = self._rerank(query, results) # _rerank uses self.top_n_rank (dynamic N)

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

        # 5. Return the context string, metadata, and the list of normalized documents
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
            "If the context does not contain relevant information use your knowledge to answer it best to knowledge "
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
            options={"temperature": 0.7, "num_ctx": 8000, "num_predict": 8000}
        )

        return response["message"]["content"]

    def query(self, question: str, chat_history: list = None, top_k: int = None, top_n: int = None):
        """
        The main synchronous entry point for the query process, now accepting dynamic parameters.
        """
        chat_history = chat_history if chat_history is not None else []

        # --- Dynamic Parameter Update (Accepting Streamlit values) ---
        # Override instance parameters if they are passed from the Streamlit UI
        if top_k is not None:
            self.top_k_retrieve = top_k
        if top_n is not None:
            self.top_n_rank = top_n
        # ---------------------------------

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
        return {"answer": answer, "sources": unique_sources, "context_chunks": documents, "raw_context_text": context}
