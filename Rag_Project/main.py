import argparse
import asyncio
import sys
from vector_db_factory import get_vector_db
from ingest_pipeline import IngestPipeline
from rag_agentic import AgenticRAG

def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument("--folder", help="Documents folder path")
    parser.add_argument("--mode", choices=["index", "query", "wipe"], required=True, help="Mode: index, query, or wipe")
    parser.add_argument("--query", help="Text query for query mode")
    args = parser.parse_args()

    collection = get_vector_db()

    if args.mode == "wipe":
        confirm = input("Are you sure you want to wipe the entire vector database? Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            collection.delete(where={})
            print("Vector database wiped.")
            sys.exit(0)
        else:
            print("Wipe cancelled.")
            sys.exit(0)

    if args.mode == "index":
        if not args.folder:
            print("Error: --folder is required in index mode.")
            sys.exit(1)
        idx = IngestPipeline()
        documents = idx.parse_docs(args.folder)
        asyncio.run(idx.index_docs(documents))

    elif args.mode == "query":
        if not args.query:
            print("Error: --query is required in query mode.")
            sys.exit(1)
        rag = AgenticRAG()
        res = rag.query(args.query)
        print("Answer:", res["answer"])
        print("Sources:")
        for src in set(res["sources"]):
            print(f"- {src}")

if __name__ == "__main__":
    main()