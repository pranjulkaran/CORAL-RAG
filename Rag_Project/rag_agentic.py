import ollama
from vector_db_factory import get_vector_db

class AgenticRAG:
    def __init__(self):
        self.collection = get_vector_db()
        self.model = "llama3.2:latest"

    def retrieve(self, query, top_k=5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results["documents"][0]
        metadata = results["metadatas"][0]
        context = "\n\n---\n\n".join(documents)
        return context, metadata

    def generate(self, query, context):
        prompt = f"""
You are a helpful assistant who answers strictly based on the context:

Context:
{context}

Question: {query}

Answer concisely with source citations.
"""
        response = ollama.generate(model=self.model, prompt=prompt, options={"temperature": 0.6, "num_ctx": 8000})
        return response["response"]

    def query(self, question):
        context, metadata = self.retrieve(question)
        answer = self.generate(question, context)
        return {"answer": answer, "sources": [md.get("source", "") for md in metadata]}
