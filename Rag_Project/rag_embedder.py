import asyncio
from concurrent.futures import ThreadPoolExecutor
import ollama

class OllamaBatchEmbedder:
    def __init__(self, model="mxbai-embed-large:335m", max_workers=4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def embed_batch(self, texts):
        loop = asyncio.get_event_loop()
        async def embed(text):
            return await loop.run_in_executor(
                self.executor,
                lambda: ollama.embeddings(model=self.model, prompt=text)['embedding']
            )
        tasks = [embed(t) for t in texts]
        return await asyncio.gather(*tasks)
