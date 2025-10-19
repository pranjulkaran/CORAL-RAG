import asyncio
from concurrent.futures import ThreadPoolExecutor
import ollama


class OllamaBatchEmbedder:
    """
    Handles concurrent, batched embedding generation using the Ollama Python client.

    Uses ThreadPoolExecutor to run blocking ollama.embeddings calls concurrently
    without blocking the main asyncio event loop, which is essential for
    fast, asynchronous ingestion.
    """

    def __init__(self, model="mxbai-embed-large:335m", max_workers=4):
        # Recommended embedding model for high-quality RAG
        # This model name should match the one available in the user's Ollama environment.
        self.model = model
        # Use a small number of workers to manage concurrent blocking calls to Ollama
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def embed_batch(self, texts):
        """
        Generates embeddings for a batch of texts concurrently.

        Args:
            texts (list[str]): A list of text strings (chunks) to embed.

        Returns:
            list[list[float]]: A list of embeddings (list of float vectors).
        """
        loop = asyncio.get_event_loop()

        # Helper function to run the blocking Ollama call in the thread pool
        async def embed(text):
            # This is the blocking call being run in the background thread
            return await loop.run_in_executor(
                self.executor,
                # The ollama.embeddings function returns a dictionary,
                # we extract just the 'embedding' vector.
                lambda: ollama.embeddings(model=self.model, prompt=text)['embedding']
            )

        # Create a task for each chunk
        tasks = [embed(t) for t in texts]

        # Run all tasks concurrently and wait for all results
        return await asyncio.gather(*tasks)
