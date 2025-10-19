import streamlit as st
import asyncio
import os
import json
from pathlib import Path
import ollama

# Assuming rag_agentic.py is in the same directory or importable
from rag_agentic import AgenticRAG

# --- Configuration & Persistence Setup ---
# Define the path for chat history persistence
PERSISTENCE_FILE = Path("chat_persistence.json")
RAG_MODE_KEYWORD = "/rag"
CHAT_MODE_KEYWORD = "/chat"


def load_chat_history():
    """Loads chat history from a local JSON file."""
    if PERSISTENCE_FILE.exists():
        try:
            with open(PERSISTENCE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load chat history: {e}")
            return []
    return []


def save_chat_history(history):
    """Saves chat history to a local JSON file."""
    try:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")


# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Agentic RAG Chat",
    layout="wide",  # Wide layout for better spacing
    initial_sidebar_state="expanded"
)


# --- Initialization ---

# Initialize AgenticRAG instance
@st.cache_resource
def get_rag_agent():
    """Initializes the RAG Agent and caches it."""
    try:
        return AgenticRAG()
    except Exception as e:
        # Check for specific errors like Ollama not running
        st.error(
            f"Failed to initialize RAG agent. Ensure Ollama is running and models are installed. Error: {e}")
        return None


# The RAG Agent object
rag_agent = get_rag_agent()

# Load initial state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "rag_mode_enabled" not in st.session_state:
    st.session_state.rag_mode_enabled = True


# --- UI and Chat Functions ---
def new_chat():
    """Clears the chat history and saves an empty state."""
    st.session_state.chat_history = []
    save_chat_history([])
    st.success("Chat history cleared. Starting new conversation!")
    st.rerun()


def handle_mode_switch(prompt):
    """Checks for mode switch keywords and updates state."""
    if prompt.lower() == CHAT_MODE_KEYWORD:
        st.session_state.rag_mode_enabled = False
        st.session_state.chat_history.append({"speaker": "System",
                                              "message": f"Switched to **Regular Chat Mode**. Database retrieval is bypassed. Type `{RAG_MODE_KEYWORD}` to return to RAG Mode."})
        return True
    elif prompt.lower() == RAG_MODE_KEYWORD:
        st.session_state.rag_mode_enabled = True
        st.session_state.chat_history.append({"speaker": "System",
                                              "message": f"Switched to **Agentic RAG Mode**. All queries will now use the vector database for grounded answers."})
        return True
    return False


def get_regular_chat_response(prompt, history):
    """Generates a simple response using Ollama without RAG context."""
    st.spinner("Generating regular chat response...")
    try:
        # Convert Streamlit history format to Ollama message format
        messages = [{"role": "system", "content": "You are a helpful, general-purpose conversational assistant."}, ]
        for msg in history:
            if msg["speaker"] == "You":
                messages.append({"role": "user", "content": msg["message"]})
            elif msg["speaker"] == "Bot":
                messages.append({"role": "assistant", "content": msg["message"]})

        # Add the new user prompt
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=rag_agent.model,  # Reuse the RAG model for consistency
            messages=messages,
            options={"temperature": 0.7, "num_ctx": 4096, "num_predict": 1000}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"An error occurred in Regular Chat Mode. Is Ollama running? Error: {e}"


# --- UI Layout ---

# Sidebar for Settings and New Chat (The "right slide" settings panel)
with st.sidebar:
    st.title("🗂️ RAG System Settings")

    current_mode_status = "🟢 RAG Mode (Database Active)" if st.session_state.rag_mode_enabled else "🔵 Chat Mode (Database Bypassed)"
    st.markdown(f"**Current Mode:** {current_mode_status}")
    st.markdown(f"Type `{CHAT_MODE_KEYWORD}` to switch mode.")
    st.markdown("---")

    # New Chat Button
    st.button("✨ Start New Chat (Clear History)", on_click=new_chat, use_container_width=True)
    st.markdown("---")

    # Display RAG Agent parameters
    st.subheader("Retrieval & Generation Params")

    if rag_agent:
        st.write(f"LLM Model: `{rag_agent.model}`")
        if hasattr(rag_agent.collection, 'name'):
            st.write(f"Vector Collection: `{rag_agent.collection.name}`")
        st.write(f"Initial Candidates (k): `{rag_agent.top_k_retrieve}`")
        st.write(f"Final Chunks (n): `{rag_agent.top_n_rank}`")
    else:
        st.warning("RAG Agent not fully initialized. Check console for errors.")

    st.markdown("---")

    st.caption("Powered by Ollama and ChromaDB.")

st.title("🔍 Agentic RAG Document Chat")
st.markdown("---")

# --- Main Chat Display Area ---

chat_container = st.container(height=500)  # Use a container to control scroll height

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        # Skip system messages in the main chat flow, display only user/bot
        if message["speaker"] == "System":
            st.info(message["message"])
            continue

        with st.chat_message(message["speaker"]):
            st.markdown(message["message"])

            # Display context chunks for the bot's RAG response
            if message["speaker"] == "Bot" and "context_chunks" in message and st.session_state.rag_mode_enabled:
                with st.expander("Show Context & Sources"):
                    # Display sources
                    if message["sources"]:
                        # Join sources and replace absolute paths with just filenames for cleanliness
                        source_names = [os.path.basename(s) for s in message["sources"]]
                        st.markdown("**Sources Used:**")
                        st.info(" | ".join(source_names))

                    # Display chunks with index
                    st.markdown("**Top Context Chunks (Re-Ranked):**")
                    for j, chunk in enumerate(message["context_chunks"]):
                        st.code(f"Chunk {j + 1}:\n{chunk}", language='text')

# --- Input Handling ---

if prompt := st.chat_input("Enter your question here..."):

    # 1. Handle mode switching keywords first
    if handle_mode_switch(prompt):
        st.rerun()

    # If the agent is not initialized, abort chat
    if not rag_agent:
        st.error("RAG agent is not initialized. Please fix the error in the sidebar/console.")
        st.rerun()

    # 2. Append user message to history and trigger rerun to display it
    st.session_state.chat_history.append({"speaker": "You", "message": prompt})
    st.rerun()

# This block executes after the user input and the immediate rerun
if st.session_state.chat_history and st.session_state.chat_history[-1]["speaker"] == "You":
    latest_prompt = st.session_state.chat_history[-1]["message"]
    # Get history excluding the current prompt
    rag_history = st.session_state.chat_history[:-1]

    bot_message = {"speaker": "Bot", "message": "", "sources": [], "context_chunks": []}

    # --- RAG Mode Execution ---
    if st.session_state.rag_mode_enabled:
        with st.spinner("🔍 Searching documents and generating RAG response..."):
            try:
                # The query method is synchronous
                result = rag_agent.query(latest_prompt, chat_history=rag_history)

                bot_message["message"] = result["answer"]
                bot_message["sources"] = result["sources"]
                bot_message["context_chunks"] = result["context_chunks"]

            except Exception as e:
                bot_message["message"] = f"An internal error occurred during RAG processing: {e}"

    # --- Regular Chat Mode Execution ---
    else:
        with st.spinner("💬 Generating regular response..."):
            try:
                answer = get_regular_chat_response(latest_prompt, rag_history)
                bot_message["message"] = answer
            except Exception as e:
                bot_message["message"] = f"An internal error occurred during regular chat: {e}"

    # 4. Append final bot response, save history, and rerun
    st.session_state.chat_history.append(bot_message)
    save_chat_history(st.session_state.chat_history)
    st.rerun()
