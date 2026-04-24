import argparse
import asyncio
import sys
import os
import traceback
import rich
import subprocess
from rich.console import Console  # Explicitly import Console for rich.print

# --- Real Imports from Project Structure ---
from vector_db_factory import get_vector_db
from ingest_pipeline import IngestPipeline
from rag_agentic import AgenticRAG

# Initialize Rich console for clean output
console = Console()


# Function to wrap the async call for use in synchronous main()
async def run_indexing(idx, documents):
    """A wrapper for the async index_docs method."""
    await idx.index_docs(documents)


def main():
    """Main function to parse arguments and execute the RAG system modes."""
    parser = argparse.ArgumentParser(
        description="A Command-Line Interface for the Agentic RAG System.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        # Added 'app' to the choices
        choices=["index", "query", "wipe", "app"],
        required=True,
        help="""\nMode of operation:
  - index: Parse and embed documents from a folder.
  - query: Retrieve and generate an answer from the indexed database.
  - wipe: Permanently delete ALL data from the vector database.
  - app: Launch the Streamlit web chat interface.
"""
    )
    parser.add_argument(
        "--folder",
        help="Path to the documents folder (required for 'index' mode)."
    )
    parser.add_argument(
        "--query",
        help="The question or text query (required for 'query' mode)."
    )
    # --- NEW ARGUMENT FOR DEBUGGING ---
    parser.add_argument(
        "--debug-prompt",
        action="store_true",  # This makes it a flag that defaults to False
        help="When used with 'query' mode, performs retrieval but skips the final LLM call, showing only the assembled prompt."
    )
    # --- END NEW ARGUMENT ---

    args = parser.parse_args()

    # --- Mode: APP ---
    if args.mode == "app":
        try:
            console.print("🌐 Launching Streamlit web application...")
            subprocess.run(["streamlit", "run", "app.py"], check=True)
        except FileNotFoundError:
            console.print("❌ Error: 'streamlit' command not found.")
            console.print(
                "Ensure Streamlit is installed (pip install streamlit) and your virtual environment is active.")
        except Exception as e:
            console.print(f"❌ An error occurred while running the Streamlit app: {e}")
            traceback.print_exc()
        sys.exit(0)


    # --- Mode: WIPE ---
    elif args.mode == "wipe":
        try:
            collection = get_vector_db()
            confirm = input(
                "❗ WARNING: Are you sure you want to wipe the entire vector database? Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                # Deleting by empty where={} deletes all documents in the collection
                collection.delete(where={})
                print("✅ Vector database completely wiped.")
            else:
                print("❌ Wipe cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error during wipe operation: {e}")
            sys.exit(1)

    # --- Mode: INDEX ---
    elif args.mode == "index":
        if not args.folder:
            print("❌ Error: --folder is required in 'index' mode.")
            sys.exit(1)

        print(f"🚀 Starting indexing pipeline for folder: {args.folder}")
        try:
            idx = IngestPipeline()

            # CRITICAL UPDATE: Call cleanup_deleted_files, passing the required folder path
            idx.cleanup_deleted_files(args.folder)

            # 2. Parse documents and check for changes
            documents = idx.parse_docs(args.folder)

            # 3. Index new documents
            if documents:
                # Correctly run the async indexing function
                asyncio.run(run_indexing(idx, documents))
                print("✅ Indexing complete.")
            else:
                print("⚠️ No new or changed documents found to index.")

        except Exception as e:
            print(f"❌ An error occurred during indexing: {e}")
            tracebox.print_exc()
            sys.exit(1)

    # --- Mode: QUERY ---
    elif args.mode == "query":
        if not args.query:
            print("❌ Error: --query is required in 'query' mode.")
            sys.exit(1)

        # New logic to determine if we are in debug mode
        is_debug_mode = args.debug_prompt

        mode_text = "DEBUG PROMPT ONLY" if is_debug_mode else "RAG GENERATION"
        console.print(
            f"🔎 Querying RAG agent with: '[bold cyan]{args.query}[/bold cyan]' (Mode: [bold]{mode_text}[/bold])")

        try:
            rag = AgenticRAG()

            # Pass the debug flag to the query method
            # This is where the magic happens: it returns the prompt instead of the answer
            res = rag.query(args.query, debug_prompt_only=is_debug_mode)

            # --- DEBUG PROMPT OUTPUT ---
            if is_debug_mode:
                print("\n" + "=" * 80)
                rich.print("[bold yellow]DEBUG PROMPT MODE: LLM GENERATION SKIPPED[/bold yellow]")
                print("=" * 80)

                print("\n📝 FULL ASSEMBLED PROMPT SENT TO LLM:")
                # Display the full prompt using rich.print for readability
                rich.print(f"[green]{res['final_prompt']}[/green]")

                print("\n📚 SOURCES USED FOR CONTEXT:")
                unique_sources = set(res["sources"])
                if unique_sources:
                    for src in unique_sources:
                        console.print(f"- [yellow]{os.path.basename(src)}[/yellow]")

                print("\n🔍 CONTEXT CHUNKS (Top Re-Ranked):")
                for i, chunk in enumerate(res["context_chunks"]):
                    # Determine source for display clarity
                    source_name = os.path.basename(res['sources'][i]) if res['sources'] and i < len(
                        res['sources']) else 'Unknown'
                    console.print(f"--- Chunk {i + 1} (Source: [cyan]{source_name}[/cyan]) ---")
                    console.print(chunk)

                print("\n" + "--------------------------------------------------")
                console.print(
                    f"To run the final LLM generation, remove the [bold yellow]'--debug-prompt'[/bold yellow] flag.")
                return  # Exit successfully after showing prompt

            # --- NORMAL RAG OUTPUT ---

            # Display formatted output
            print("\n" + "=" * 50)
            print("🤖 Answer:")
            print(res["answer"])
            print("=" * 50)

            print("\n📚 Sources Used:")
            unique_sources = set(res["sources"])
            if unique_sources:
                for src in unique_sources:
                    # Print just the filename for cleaner output
                    print(f"- {os.path.basename(src)}")
            else:
                print("- None (Answer may be based on common LLM knowledge or context was empty.)")

            print("\n🔍 Context Chunks (Top Re-Ranked):")
            for i, chunk in enumerate(res["context_chunks"]):
                # Simple display of the actual chunks used
                print(f"--- Chunk {i + 1} ---")
                print(chunk)

            print("--------------------------------------------------")

        except Exception as e:
            print(f"❌ An error occurred during query: {e}")
            # print the traceback for easier debugging if a new error occurs
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
