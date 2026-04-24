import re
import random
from typing import List
import os

# You must install the ChromaDB client: pip install chromadb
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

# --- 1. CONFIGURATION ---
# >>>>>> SYNCHRONIZED WITH YOUR CONFIGURATION <<<<<<

# The path where ChromaDB stores its persistent files (from your CHROMA_DB_PATH)
CHROMA_DB_PATH = r"D:\rag_storage"

# The name of your Chroma collection (Synchronized with your get_vector_db() function)
CHROMA_COLLECTION_NAME = "rag_docs"

# The number of chunks to randomly sample and audit.
SAMPLE_SIZE = 100000

# --- 2. Quality Checker Heuristics ---

class QualityChecker:
    """
    Applies structural integrity checks to text chunks.
    Chunks that fail these checks are considered 'bad' or noisy.
    """
    def __init__(self, min_length: int = 50, max_punctuation_ratio: float = 0.30):
        # Chunks shorter than this minimum character count are considered too fragmented.
        self.MIN_LENGTH = min_length
        # Chunks where non-standard punctuation/noise exceeds this ratio are flagged.
        self.MAX_PUNCTUATION_RATIO = max_punctuation_ratio

    def is_chunk_bad(self, text: str) -> bool:
        """
        Runs all heuristics against a single text chunk.
        Returns True if the chunk should be filtered out.
        """
        text = text.strip()

        # Heuristic 1: Minimum Length Check
        if len(text) < self.MIN_LENGTH:
            return True

        # Heuristic 2: Punctuation/Noise Ratio Check
        total_chars = len(text)
        if total_chars == 0:
            return True

        # This regex pattern counts characters that are NOT word characters, whitespace, or
        # standard punctuation (.,:;()-). This targets excessive boilerplate, separators, or merged noise.
        # This is a critical check for Markdown/PDF extraction noise.
        noise_chars = len(re.findall(r'[^\w\s\.\,\:\;\(\)\-]', text, re.UNICODE))

        punctuation_ratio = noise_chars / total_chars
        if punctuation_ratio > self.MAX_PUNCTUATION_RATIO:
            return True

        # Passed all checks
        return False

# --- 3. Chroma Retrieval Logic ---

def retrieve_random_sample_from_chroma(collection: Collection, sample_size: int) -> List[str]:
    """
    Retrieves a random sample of chunk texts from the Chroma collection.
    It retrieves a batch using peek and then randomizes the selection in memory.
    """
    count = collection.count()
    if count == 0:
        print(f"Error: Collection '{CHROMA_COLLECTION_NAME}' is empty.")
        return []

    # Strategy: Retrieve up to 2x the sample size, then randomize,
    # as Chroma doesn't have a native 'get_random' function.
    limit_for_peek = min(count, sample_size * 2)

    print(f"Total documents in collection: {count:,}")
    print(f"Attempting to peek {limit_for_peek:,} documents for randomization...")

    try:
        # Retrieve the documents
        results = collection.peek(limit=limit_for_peek)

        # Filter out None/empty documents and flatten the list of texts
        all_texts = [doc for doc in results.get('documents', []) if doc is not None and doc.strip()]

        # Randomly select the required sample size
        final_sample_size = min(sample_size, len(all_texts))
        if final_sample_size == 0:
             print("Error: Peek operation returned no valid document texts.")
             return []

        random_sample = random.sample(all_texts, final_sample_size)

        print(f"Successfully retrieved and sampled {len(random_sample)} chunks for audit.")
        return random_sample

    except Exception as e:
        print(f"An error occurred during Chroma retrieval: {e}")
        return []


# --- 4. Main Audit Function ---

def run_audit():
    """
    Connects to Chroma via PersistentClient, retrieves a sample, runs the checker, and calculates the percentage.
    """
    print(f"Connecting to Chroma Persistent DB at: {CHROMA_DB_PATH}")

    # Initialize the Chroma Persistent Client
    try:
        # Using PersistentClient to connect to the local directory
        # Anonymized telemetry is set to False, matching your setup's spirit
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))
        # Get the collection (using get_collection, as we assume it already exists from your ingestion)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    except Exception as e:
        print(f"Failed to connect to Chroma or retrieve collection: {e}")
        print(f"Ensure the path '{CHROMA_DB_PATH}' is correct, and the collection '{CHROMA_COLLECTION_NAME}' exists.")
        return

    # Get the sample data
    sample_chunks = retrieve_random_sample_from_chroma(collection, SAMPLE_SIZE)

    if not sample_chunks:
        return

    total_chunks = len(sample_chunks)
    checker = QualityChecker()
    bad_chunks_count = 0

    # Run the audit
    for chunk in sample_chunks:
        if checker.is_chunk_bad(chunk):
            bad_chunks_count += 1

    # --- 5. Calculate and Display Results ---

    good_chunks_count = total_chunks - bad_chunks_count
    print("\n" + "="*60)
    print(f"| RAG Vector Store Quality Audit")
    print("="*60)
    print(f"| DB Path: {CHROMA_DB_PATH}")
    print(f"| Collection: {CHROMA_COLLECTION_NAME}")
    print(f"| Heuristics: Min Length {checker.MIN_LENGTH}, Max Noise Ratio {checker.MAX_PUNCTUATION_RATIO}")
    print("-"*60)
    print(f"| Total Chunks Audited (Sample): {total_chunks:,}")
    print(f"| Good Chunks (Passed Checks):   {good_chunks_count:,}")
    print(f"| Bad/Noisy Chunks (Failed Checks): {bad_chunks_count:,}")
    print("-"*60)

    if total_chunks > 0:
        noise_percentage = (bad_chunks_count / total_chunks) * 100
        print(f"| Calculated Noise Percentage:   {noise_percentage:.2f}%")
        print("="*60)

        # Recommendation based on thresholds
        if noise_percentage >= 15.0:
            print("\n🚨 RECOMMENDATION: NOISE IS HIGH (>= 15.0%).")
            print("Action: **Urgent implementation of a cleaning/filtering** step is needed during ingestion.")
        elif noise_percentage >= 5.0:
            print("\n⚠️ RECOMMENDATION: NOISE IS MODERATE (5.0% - 15.0%).")
            print("Action: Implement **Cleaning/Filtering** to boost retrieval quality and reduce LLM hallucinations.")
        else:
            print("\n✅ RECOMMENDATION: NOISE IS LOW (< 5.0%).")
            print("Action: You're in a good state. Continue to monitor, or focus on advanced chunking strategies.")
    else:
        print("No chunks were retrieved to perform the calculation.")

if __name__ == "__main__":
    run_audit()
