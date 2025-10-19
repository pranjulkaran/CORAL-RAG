import streamlit as st
import asyncio
# Assuming rag_agentic.py is in the same directory or importable
from rag_agentic import AgenticRAG

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Agentic RAG Chat",
    layout="wide",  # Use wide layout for better spacing (addressing "more right")
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
        # Check for specific errors like collection not found (assuming the fix was applied)
        st.error(
            f"Failed to initialize RAG agent. Please ensure Ollama is running, models are installed, and the vector store is correctly configured. Error: {e}")
        return None


rag_agent = get_rag_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- New Chat Functionality ---
def new_chat():
    """Clears the chat history."""
    st.session_state.chat_history = []
    st.success("Chat history cleared. Starting new conversation!")
    # FIX: Use st.rerun()
    st.rerun()


# --- UI Layout ---

# Sidebar for Settings and New Chat (The "right slide" settings panel)
with st.sidebar:
    st.title("🗂️ RAG Settings")

    # New Chat Button
    st.button("✨ Start New Chat", on_click=new_chat, use_container_width=True)
    st.markdown("---")

    # Display RAG Agent parameters (read from the imported class)
    st.subheader("Retrieval & Generation Params")

    if rag_agent:
        st.write(f"Model: `{rag_agent.model}`")
        # Ensure collection attribute exists before trying to access .name
        if hasattr(rag_agent.collection, 'name'):
            st.write(f"Collection: `{rag_agent.collection.name}`")
        st.write(f"Candidates (k): `{rag_agent.top_k_retrieve}`")
        st.write(f"Final Chunks (n): `{rag_agent.top_n_rank}`")
    else:
        st.write("RAG Agent not fully initialized.")

    st.markdown("---")

    st.caption("Powered by Ollama, ChromaDB, and Agentic RAG logic.")

st.title("🔍 Agentic RAG Document Chat")
st.markdown("Ask a question about the documents indexed in your vector store.")

# --- Main Chat Display Area ---

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    # Use st.chat_message for standard Streamlit chat bubbles
    with st.chat_message(message["speaker"]):
        st.markdown(message["message"])

        # Display context chunks for the bot's response
        if message["speaker"] == "Bot" and "context_chunks" in message and message["context_chunks"]:
            with st.expander("Show Context & Sources"):
                # Display sources
                if message["sources"]:
                    st.markdown("**Sources Used:**")
                    st.info(", ".join(message["sources"]))

                # Display chunks with index
                st.markdown("**Top Context Chunks (Re-Ranked):**")
                for j, chunk in enumerate(message["context_chunks"]):
                    st.code(f"Chunk {j + 1}:\n{chunk}", language='text')

# --- Input Handling ---

if prompt := st.chat_input("Enter your question here..."):

    # Check if agent initialized before proceeding
    if not rag_agent:
        st.session_state.chat_history.append({"speaker": "You", "message": prompt})
        st.error("RAG agent is not initialized. Please fix the error in the sidebar/console.")
        # FIX: Use st.rerun()
        st.rerun()

    # 1. Display user message
    st.session_state.chat_history.append({"speaker": "You", "message": prompt})

    # FIX: Use st.rerun()
    st.rerun()

# This part runs after the immediate rerun triggered by the user input
if st.session_state.chat_history and st.session_state.chat_history[-1]["speaker"] == "You":
    # 2. Get chat history subset for RAG (excluding the latest user prompt, as it's processed separately)
    latest_prompt = st.session_state.chat_history[-1]["message"]

    # We pass the full history (excluding the latest, unprocessed user message) to the RAG agent
    # for conversational context.
    rag_history = st.session_state.chat_history[:-1]

    # 3. Generate response using the RAG agent
    with st.spinner("Searching documents and generating response..."):
        try:
            # The query method is synchronous
            result = rag_agent.query(latest_prompt, chat_history=rag_history)

            # The result object contains 'answer', 'sources', and 'context_chunks'
            answer = result["answer"]
            sources = result["sources"]
            context_chunks = result["context_chunks"]

        except Exception as e:
            answer = f"An internal error occurred during RAG processing: {e}"
            sources = []
            context_chunks = []

    # 4. Append bot response (replace the temporary 'You' entry with the full Bot response)
    bot_message = {
        "speaker": "Bot",
        "message": answer,
        "sources": sources,
        "context_chunks": context_chunks
    }
    # Append the bot message to history
    st.session_state.chat_history.append(bot_message)

    # FIX: Use st.rerun()
    st.rerun()
