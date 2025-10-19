import ollama
import asyncio
from vector_db_factory import get_vector_db
from rag_embedder import OllamaBatchEmbedder  # Assumes this class name is correct


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
        self.embedder = OllamaBatchEmbedder()

        # LLM to be used for final answer generation
        # FIXED to use the installed model 'llama3.2:latest'
        self.model = "llama3.2:latest"

        # Configuration for Two-Stage Retrieval
        self.top_k_retrieve = 15  # Stage 1: Initial number of candidates retrieved from Vector DB
        self.top_n_rank = 5  # Stage 2: Final number of best chunks passed to the LLM (Re-Ranked subset)

    def _rerank(self, query: str, results: dict):
        """
        Simulates the re-ranking step. Since we don't use a dedicated cross-encoder
        here, we rely on Chroma's similarity score ordering and simply truncate
        the results to the top_n_rank.
        """

        # ChromaDB query results are lists nested inside another list (e.g., [[]])
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
        query_embedding_list = await self.embedder.embed_batch([query])
        query_embedding = query_embedding_list[0]

        # 2. Query the vector store for a large number of candidate chunks (Stage 1)
        results = self.collection.query(
            query_embeddings=[query_embedding],  # CRITICAL: Use query_embeddings with the Ollama vector
            n_results=self.top_k_retrieve,  # Retrieve the larger candidate set
            include=['documents', 'metadatas', 'distances']
        )

        if not results.get("documents") or not results["documents"][0]:
            return "", [], []

        # 3. Apply Re-Ranking/Filtering to get the best N chunks (Stage 2)
        documents, metadata, _ = self._rerank(query, results)

        # Create context string from re-ranked documents
        context = "\n\n---\n\n".join(documents)
        return context, metadata, documents

    def generate(self, query: str, context: str, chat_history: list):
        """
        Generates the final answer using the LLM based on the retrieved context.
        """
        # Format the conversational history for the LLM prompt
        history_str = "\n".join([f"{h['speaker']}: {h['message']}" for h in chat_history])

        # 🚨 FINALIZED SYSTEM PROMPT: Stronger directive and explicit instruction to avoid hedging. 🚨
        system_prompt = (
            "You are a helpful assistant. Use the provided CONTEXT to formulate your answer. "
            "If the context contains information that directly or indirectly answers the question, summarize and state it clearly. "
            "**Do not state that the information is 'inferred' or 'not explicitly mentioned' if the components are present in the text.** "
            "If the context does not contain relevant information, state that you cannot answer based on the provided documents. "
            "Always include source citations at the end of your answer."
        )

        # 🚨 FINALIZED USER MESSAGE: Guiding the LLM to look for specific components. 🚨
        user_message_content = (
            f"HISTORICAL CONVERSATION:\n{history_str}\n\n"
            f"NEW CONTEXT (Use this for your answer, this context has been pre-filtered for relevance):\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Please identify the main components or methods of the Redux store object from the CONTEXT and list them as the answer."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.6, "num_ctx": 8000}
        )

        return response["message"]["content"]

    def query(self, question: str, chat_history: list = None):
        """
        The main synchronous entry point for the query process.
        """
        chat_history = chat_history if chat_history is not None else []

        # Use asyncio.run to execute the async retrieval function
        try:
            context, metadata, documents = asyncio.run(self.retrieve(question))
        except Exception as e:
            # Handle retrieval errors gracefully
            print(f"Error during async retrieval: {e}")
            return {"answer": f"An unexpected error occurred during context retrieval: {e}", "sources": [],
                    "context_chunks": []}

        if not context:
            return {"answer": "I could not find any relevant documents in the database to answer your question.",
                    "sources": [], "context_chunks": []}

        answer = self.generate(question, context, chat_history)

        # Extract unique source names (file paths)
        unique_sources = list(set(md.get("source", "Unknown Source") for md in metadata))

        # The documents list here only contains the final, re-ranked chunks (top_n_rank)
        return {"answer": answer, "sources": unique_sources, "context_chunks": documents}
