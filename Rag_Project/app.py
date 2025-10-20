import streamlit as st
import asyncio
import os
import json
from pathlib import Path
import ollama
import time

# NOTE: The rag_agentic module must be available in the environment to run this app.
# Assuming 'rag_agentic' is accessible in the environment.
from rag_agentic import AgenticRAG

# --- RAG Parameter Defaults ---
DEFAULT_TOP_K = 15
DEFAULT_TOP_N = 5

# --- Configuration & Persistence Setup ---
# Define the path for chat history persistence
PERSISTENCE_FILE = Path("chat_persistence.json")
RAG_MODE_KEYWORD = "/rag"
CHAT_MODE_KEYWORD = "/chat"

# --- Inject Custom Dark Theme CSS ---
CUSTOM_CSS = """
<style>
/* 1. Global Background, Text, and Font */
.stApp {
    background-color: #0d0d0d; /* DEEPER DARK background */
    color: #c0c0c0; /* Slightly darker text */
    font-size: 14px; /* Smaller base font */
    font-family: 'Inter', sans-serif;
}

/* 2. Main Content and Sidebar Backgrounds */
.main, .stSidebar > div:first-child {
    background-color: #0d0d0d;
}

/* --- NEW: Fixed Width and Center Alignment --- */
/* Target the main content container and restrict its width */
.main > div {
    max-width: 900px !important; /* Fixed width for centering */
    margin: 0 auto; /* Center alignment */
    padding-left: 0;
    padding-right: 0;
}

/* Ensure Streamlit's internal blocks also respect the max-width */
[data-testid="stVerticalBlock"] {
    max-width: 900px; 
    margin: 0 auto;
}

/* Ensure the chat input area aligns correctly at the bottom */
div[data-testid="stForm"] {
    max-width: 900px;
    margin: 0 auto;
    padding-top: 10px;
}
/* ------------------------------------------- */

/* 3. Container and Panel Backgrounds (primary-panel) - Used for st.containers and alerts */
.stChatMessage, .stContainer, .stAlert > div {
    background-color: #1a1a1a !important; /* Darkened primary panels */
    border-radius: 8px; /* Slightly smaller radius */
    border: 1px solid #333333; /* Darker, less noticeable border */
    box-shadow: none; /* Removed heavy shadow for cleaner look */
}

/* 4. Chat Bubbles */
/* Bot/System messages (Left aligned, Dark background) */
.stChatMessage div[data-testid="stChatMessageContent"] {
    background-color: #222222 !important; 
    color: #c0c0c0;
    font-size: 14px; 
    padding: 10px 14px; 
    border-radius: 0.5rem 0.5rem 0.5rem 0; 
}

/* User messages (Right aligned, Muted Slate Blue: #406180) */
.stChatMessage.st-cm-user div[data-testid="stChatMessageContent"] {
    background-color: #406180 !important; 
    color: white;
    font-size: 14px; 
    padding: 10px 14px; 
    border-radius: 0.5rem 0.5rem 0 0.5rem; 
}

/* 5. Inputs, Selectors, and Text Areas */
div[data-testid="stForm"] .stTextInput input, div[data-testid="stForm"] .stTextArea textarea, 
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {
    background-color: #111111; 
    border: 1px solid #333333; 
    color: #c0c0c0;
    border-radius: 6px;
    padding: 6px 10px; 
    font-size: 14px;
}

/* 6. Custom Scrollbar for Chat Container Only */
/* The st.container(height=...) ensures vertical scroll is only here */
.main-container .stChat, .main-container .stSidebar {
    scrollbar-width: thin;
    scrollbar-color: #333333 #1a1a1a;
}

/* 7. Button Customization */
.stButton>button {
    background-color: #406180; 
    color: white;
    border-radius: 6px;
    border: none;
    transition: background-color 0.2s, transform 0.2s;
}

.stButton>button:hover {
    background-color: #314a63; 
    transform: scale(1.01);
}

/* 8. Radio/Checkbox Color - Muted Green for active status */
.stRadio > label > div:first-child, .stCheckbox > label > div:first-child {
    border-color: #10b981 !important; 
}
.stRadio > label > div:first-child > div, .stCheckbox > label > div:first-child > div {
    background-color: #10b981 !important; 
}

/* 9. st.code blocks */
div[data-testid="stCodeBlock"] pre {
    background-color: #111111;
    border: 1px solid #333333;
    color: #a0a0a0;
    font-size: 13px;
}

/* Sidebar Specifics for Compactness */
.stSidebar h2, .stSidebar h3 {
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 1.1rem; 
}
.stSidebar .stMarkdown p {
    font-size: 0.85rem; 
}
.stSidebar .stRadio > label {
    font-size: 0.9rem; 
    margin-bottom: 0.1rem; 
}

/* Toning down colorful st.info/st.warning alerts */
div[data-testid="stAlert"] {
    font-size: 14px;
}
div[data-testid="stAlert-info"], div[data-testid="stAlert-success"], div[data-testid="stAlert-warning"], div[data-testid="stAlert-error"] {
    background-color: #1a1a1a !important; 
    color: #c0c0c0 !important; 
    border-color: #333333 !important;
}

</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Data Handling Functions ---

def load_chat_history():
    """Loads chat history from a local JSON file."""
    if PERSISTENCE_FILE.exists():
        try:
            with open(PERSISTENCE_FILE, 'r') as f:
                # Ensure the loaded data is a list if the file is not empty/corrupted
                data = json.load(f)
                return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            print(f"Chat history file is corrupted or empty at: {PERSISTENCE_FILE}. Starting new history.")
            return []
        except Exception as e:
            print(f"Failed to load chat history: {e}")
            return []
    return []


def save_chat_history(history):
    """Saves chat history to a local JSON file."""
    try:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Failed to save chat history: {e}")


# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Agentic RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)


# --- Initialization ---

# Initialize AgenticRAG instance
@st.cache_resource
def get_rag_agent():
    """Initializes the RAG Agent and caches it."""
    try:
        # Check if Ollama is accessible
        ollama.list()
        st.info("Ollama server connected successfully.")
        return AgenticRAG()
    except Exception as e:
        st.error(
            f"Failed to initialize RAG agent. Ensure **Ollama** is running and the required models are installed (e.g., `llama3`). Error: {e}")
        return None


# The RAG Agent object
rag_agent = get_rag_agent()

# Load initial state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "rag_mode_enabled" not in st.session_state:
    st.session_state.rag_mode_enabled = True

# Initialize RAG parameters in session state
if "top_k_retrieve" not in st.session_state:
    st.session_state.top_k_retrieve = DEFAULT_TOP_K
if "top_n_rank" not in st.session_state:
    st.session_state.top_n_rank = DEFAULT_TOP_N


# --- UI and Chat Functions ---
def new_chat():
    """Clears the chat history and saves an empty state."""
    st.session_state.chat_history = []
    save_chat_history([])
    st.success("Chat history cleared. Starting new conversation!")
    st.rerun()


def handle_mode_switch(prompt):
    """Checks for mode switch keywords and updates state."""
    # Note: Mode switch commands are case-insensitive
    if prompt.lower() == CHAT_MODE_KEYWORD:
        st.session_state.rag_mode_enabled = False
        st.session_state.chat_history.append({"speaker": "System",
                                              "message": f"Switched to **Regular Chat Mode** (LLM Only). Database retrieval is bypassed. Type `{RAG_MODE_KEYWORD}` to return to grounded chat."})
        return True
    elif prompt.lower() == RAG_MODE_KEYWORD:
        st.session_state.rag_mode_enabled = True
        st.session_state.chat_history.append({"speaker": "System",
                                              "message": f"Switched to **Agentic RAG Mode** (Grounded). All queries will now use the vector database. Type `{CHAT_MODE_KEYWORD}` to switch to LLM only chat."})
        return True
    return False


def get_regular_chat_response(prompt, history):
    """Generates a simple response using Ollama without RAG context."""
    st.spinner("Generating regular chat response...")
    # Convert Streamlit history format to Ollama message format
    messages = [{"role": "system", "content": "You are a helpful, general-purpose conversational assistant."}, ]

    # Add previous conversation context
    for msg in history:
        if msg["speaker"] == "You":
            messages.append({"role": "user", "content": msg["message"]})
        elif msg["speaker"] == "Bot":
            # Assuming the bot's previous message is the answer, not the whole structured message
            messages.append({"role": "assistant", "content": msg["message"]})

    # Add the new user prompt
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=rag_agent.model,  # Reuse the RAG model for consistency
        messages=messages,
        options={"temperature": 0.7, "num_ctx": 4096, "num_predict": 1000}
    )
    return response["message"]["content"]


@st.cache_data(show_spinner=False)
def get_db_count(_rag_agent):
    """
    Fetches the document count from the vector database.
    """
    if _rag_agent and _rag_agent.collection:
        try:
            return _rag_agent.collection.count()
        except Exception:
            return -1
    return 0


# --- UI Layout ---

# Sidebar for Settings and New Chat (The "right slide" settings panel)
with st.sidebar:
    st.title("🗂️ RAG System Settings")
    st.markdown("<p style='color:#a0a0a0; font-size: 0.85rem;'>Customize your retrieval and generation parameters.</p>",
                unsafe_allow_html=True)

    # --- DATABASE STATUS UI ---
    st.subheader("Database Status")
    db_count = get_db_count(rag_agent)

    if db_count > 0:
        st.markdown(f"<p style='color:#10b981; font-weight: bold;'>✨ Indexed Chunks: {db_count:,}</p>",
                    unsafe_allow_html=True)
        st.caption("System is grounded in your private documents.")
    elif db_count == 0:
        st.warning("Database is empty. Run `python main.py --mode index` to ingest documents.")
    else:
        st.error("Database connection failed. Check your ChromaDB configuration.")
    st.markdown("---")
    # --- END DATABASE STATUS UI ---

    # --- CONVERSATION MODE SELECTOR ---
    st.subheader("Conversation Mode")

    # Set the initial selection based on the current session state
    initial_index = 0 if st.session_state.rag_mode_enabled else 1

    selected_mode = st.radio(
        "Select Interaction Mode:",
        ("Agentic RAG (Local Documents)", "Regular Chat (LLM Only)"),
        index=initial_index,
    )

    # Update the session state based on the radio button's selection
    if selected_mode == "Agentic RAG (Local Documents)":
        st.session_state.rag_mode_enabled = True
        st.caption(
            f"🟢 **RAG Mode** is active. Retrieves documents for grounded answers. Use command `{CHAT_MODE_KEYWORD}` to switch.")
    else:
        st.session_state.rag_mode_enabled = False
        st.caption(
            f"🔵 **Chat Mode** is active. General conversation, bypassing the database. Use command `{RAG_MODE_KEYWORD}` to switch.")

    st.markdown("---")

    # --- RAG Parameter Settings (COMPACT/INLINE) ---
    st.subheader("Retrieval Tuning (K and N)")
    st.caption("Adjust the depth of search and final context size for RAG mode.")

    # Use columns to put K and N side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top K (Retrieve)**",
                    help="Initial number of most similar chunks to retrieve from the vector database (Candidate Generation).")
        # Top K Retrieval: How many documents to fetch initially
        new_top_k = st.number_input(
            "Top K Candidates (Retrieval Depth)",
            min_value=5,
            max_value=100,
            value=st.session_state.top_k_retrieve,
            step=5,
            key="top_k_input",
            label_visibility="collapsed"  # Hides the default label for compactness
        )
    with col2:
        st.markdown("**Top N (Context)**",
                    help="Final number of best chunks to send to the LLM for generation (Re-Ranked Subset).")
        # Top N Rank: How many documents to keep after re-ranking
        new_top_n = st.number_input(
            "Top N Chunks (Context Size)",
            min_value=1,
            max_value=new_top_k,  # N <= K constraint applied here
            value=st.session_state.top_n_rank,
            step=1,
            key="top_n_input",
            label_visibility="collapsed"  # Hides the default label for compactness
        )

    # Enforce N <= K constraint
    if new_top_n > new_top_k:
        new_top_n = new_top_k

    st.session_state.top_k_retrieve = new_top_k  # Update K state
    st.session_state.top_n_rank = new_top_n  # Update N state

    st.markdown("---")
    # --- END RAG PARAMETER SETTINGS ---

    # New Chat Button
    st.button("✨ Start New Chat (Clear History)", on_click=new_chat, use_container_width=True)
    st.markdown("---")

    # Display Active RAG Agent parameters
    st.subheader("Active Agent Status")

    if rag_agent:
        st.markdown(f"**LLM Model:** `{rag_agent.model}`")
        if hasattr(rag_agent.collection, 'name'):
            st.markdown(f"**Vector Collection:** `{rag_agent.collection.name}`")
        # Display the *currently active* values from session state
        st.markdown(f"**Current K:** `{st.session_state.top_k_retrieve}`")
        st.markdown(f"**Current N:** `{st.session_state.top_n_rank}`")
    else:
        st.warning("RAG Agent not fully initialized. Check the connection errors above.")

    st.markdown("---")

    st.caption("Powered by **Ollama** and **ChromaDB** (Local RAG).")

# --- Main Chat Display Area (Now just a container for history and input) ---

# Use a fixed height container for the chat history
# Note: The CSS ensures this entire main content block is centered and fixed width.
chat_container = st.container(height=550, border=True)

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:

        # System messages display outside chat bubbles for announcements
        if message["speaker"] == "System":
            st.info(message["message"])
            continue

        # Use st.chat_message for the standard bubble look
        with st.chat_message(message["speaker"]):
            st.markdown(message["message"])

            # Display context chunks for the bot's RAG response
            if message["speaker"] == "Bot" and "context_chunks" in message and st.session_state.rag_mode_enabled:
                with st.expander("Show Context & Sources"):
                    # Display sources
                    if message["sources"]:
                        st.markdown("**Sources Used:**")
                        source_display = []
                        for s in message["sources"]:
                            # Source is a local file path (e.g., 'data/document.pdf')
                            source_display.append(f"- `{os.path.basename(s)}`")
                        st.markdown("\n".join(source_display), unsafe_allow_html=True)

                    # Display chunks with index
                    if message["context_chunks"]:
                        st.markdown("**Top Context Chunks (Re-Ranked):**")
                        for j, chunk in enumerate(message["context_chunks"]):
                            # Using st.code for clear chunk display
                            st.code(f"Chunk {j + 1}:\n{chunk}", language='text')

# --- Input Handling ---

# Determine the placeholder text based on the current mode
input_placeholder = (
    f"Type your grounded question or `{CHAT_MODE_KEYWORD}` to switch to LLM only..."
    if st.session_state.rag_mode_enabled
    else f"Ask a general question or `{RAG_MODE_KEYWORD}` to switch back to RAG..."
)

if prompt := st.chat_input(input_placeholder):

    # 1. Handle mode switching keywords first
    if handle_mode_switch(prompt):
        save_chat_history(st.session_state.chat_history)
        st.rerun()

    # If the RAG agent failed initialization, prevent chat
    if not rag_agent:
        st.error("RAG agent is not initialized. Please ensure Ollama is running and configured correctly.")
        st.stop()

    # 2. Append user message to history and trigger rerun to display it immediately
    st.session_state.chat_history.append({"speaker": "You", "message": prompt})
    save_chat_history(st.session_state.chat_history)
    st.rerun()

# This block executes after the user input and the immediate rerun, ensuring the bot responds
if (st.session_state.chat_history and
        st.session_state.chat_history[-1]["speaker"] == "You" and
        st.session_state.chat_history[-1]["message"].lower() not in [CHAT_MODE_KEYWORD, RAG_MODE_KEYWORD]):

    latest_prompt = st.session_state.chat_history[-1]["message"]
    rag_history = st.session_state.chat_history[:-1]

    bot_message = {"speaker": "Bot", "message": "", "sources": [], "context_chunks": []}

    # --- RAG Mode Execution ---
    if st.session_state.rag_mode_enabled:
        with st.spinner("🔍 Searching local documents and generating RAG response..."):
            try:
                # 1. Attempt local RAG query
                # --- CRITICAL: PASSING DYNAMIC K and N values ---
                result = rag_agent.query(
                    latest_prompt,
                    chat_history=rag_history,
                    top_k=st.session_state.top_k_retrieve,
                    top_n=st.session_state.top_n_rank
                )

                # Check if local RAG was successful (found documents)
                if result and result.get("context_chunks"):
                    bot_message["message"] = result["answer"]
                    bot_message["sources"] = result.get("sources", [])
                    bot_message["context_chunks"] = result.get("context_chunks", [])

                # Local RAG Fails: Couldn't find relevant context
                else:
                    bot_message["message"] = (
                        "I could not find any relevant documents in the local database to confidently answer your question. "
                        "Try a different phrasing or consider switching to **Regular Chat Mode** (`/chat`) for a general LLM response."
                    )

            except Exception as e:
                # General error in RAG process
                print(f"RAG Processing Error: {e}")
                bot_message["message"] = f"An internal error occurred during RAG processing. Details: {e}"

    # --- Regular Chat Mode Execution ---
    else:
        with st.spinner("💬 Generating regular response..."):
            try:
                answer = get_regular_chat_response(latest_prompt, rag_history)
                bot_message["message"] = answer
            except Exception as e:
                print(f"Regular Chat Error: {e}")
                bot_message["message"] = f"An internal error occurred during regular chat. Details: {e}"

    # 4. Append final bot response, save history, and rerun
    st.session_state.chat_history.append(bot_message)
    save_chat_history(st.session_state.chat_history)
    st.rerun()
