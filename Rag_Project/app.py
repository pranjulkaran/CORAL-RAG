import streamlit as st
from rag_agentic import AgenticRAG
import os
import copy

# Set page configuration for wide layout and title
st.set_page_config(page_title="Agentic RAG Chat", layout="wide", page_icon="🤖")
st.title("Retrieval-Augmented Generation Chat")

# Inject custom CSS for dark chat bubbles and better readability
st.markdown("""
<style>
/* Base Streamlit Container and Text Color (Assuming system dark mode) */
body {
    color: white; /* Ensure default text is readable */
}

/* Custom styling for User Message */
.user-message {
    background-color: #34495e; /* Deep Slate */
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 15px;
    text-align: left;
    border-left: 5px solid #2980b9; /* Bright Blue accent */
}

/* Custom styling for Bot Message */
.bot-message {
    background-color: #2c3e50; /* Deep Gray/Blue */
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 5px;
    border-left: 5px solid #27ae60; /* Green accent */
}

/* Adjust the style for the code blocks inside the expander */
div.streamlit-expanderContent code {
    background-color: #1f2a38; /* Slightly darker background for code */
    border-radius: 5px;
    padding: 10px;
    display: block;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)


# --- Initialisation and Caching ---
@st.cache_resource(show_spinner=False)
def get_rag_instance():
    with st.spinner("🚀 Setting up RAG agent and Vector DB..."):
        return AgenticRAG()


# Ensure the agent is initialized once
try:
    rag = get_rag_instance()
except Exception as e:
    st.error(f"Failed to initialize RAG agent. Ensure Ollama is running and models are installed: {e}")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("RAG Settings")

    # Control the final number of RERANKED chunks (top_n_rank)
    top_n_rank_value = st.slider(
        "Final Context Chunks (Top N after Re-Rank)",
        min_value=3,
        max_value=15,
        value=rag.top_n_rank,  # Use the default from the agent
        step=1,
        help=f"The number of *most relevant* chunks passed to the LLM. The system initially retrieves {rag.top_k_retrieve} candidates."
    )

    # Apply the slider value to the agent instance
    rag.top_n_rank = top_n_rank_value

    history_size = st.slider(
        "Conversation History Turns",
        min_value=0,
        max_value=10,
        value=4,
        step=2,
        help="Number of previous Q/A pairs to send for conversational context."
    )

    st.markdown("---")
    st.info(f"LLM Model: `{rag.model}`\nEmbed Model: `mxbai-embed-large`")
    if st.button("Clear Chat History", key="clear_chat_button"):
        st.session_state.chat_history = []
        st.rerun()


def on_ask():
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Append user question to full history (so it appears immediately)
        st.session_state.chat_history.append(
            {"speaker": "You", "message": user_input, "sources": [], "context_chunks": []})

        with st.spinner("🤖 Thinking..."):
            try:
                # --- Extract Conversational History ---
                history_turns = []
                # Build history list, excluding the current user message (added above)
                for h in st.session_state.chat_history[:-1]:
                    if h["sources"] or h["speaker"] == "You":
                        history_turns.append({"speaker": h["speaker"], "message": h["message"]})

                history_for_rag = history_turns[-history_size:]

                # --- RAG Query (No longer passes top_k) ---
                # CRITICAL: The call no longer passes top_k
                result = rag.query(user_input, chat_history=history_for_rag)

                # Append bot answer to full history
                st.session_state.chat_history.append({
                    "speaker": "Bot",
                    "message": result["answer"],
                    "sources": result.get("sources", []),
                    "context_chunks": result.get("context_chunks", [])
                })

            except Exception as e:
                # Display error message in the chat history
                error_message = f"An error occurred while querying the RAG system: {e}"
                st.session_state.chat_history.append(
                    {"speaker": "Bot", "message": error_message, "sources": [], "context_chunks": []})

            st.session_state.user_input = ""


# --- Chat Input Widget ---
st.text_input("Ask your question:", key="user_input", on_change=on_ask,
              placeholder=f"Ask a question using the top {rag.top_n_rank} re-ranked chunks...")

# --- Chat History Display ---
st.markdown("---")

chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        speaker = chat["speaker"]
        message = chat["message"]
        sources = chat["sources"]
        context_chunks = chat["context_chunks"]

        if speaker == "You":
            st.markdown(
                f'<div class="user-message"><b>You:</b> {message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message"><b>Bot:</b> {message}</div>',
                unsafe_allow_html=True
            )

            if context_chunks:
                with st.expander(f"📚 Retrieval Details (Used {len(context_chunks)} Final Chunks after Re-Ranking)"):

                    st.subheader("Source Documents")
                    source_list_markdown = ""
                    for s in sources:
                        filename = os.path.basename(s)
                        source_list_markdown += f"- **{filename}** (`{s}`)\n"
                    st.markdown(source_list_markdown)

                    st.subheader("Final Context Chunks Sent to LLM")
                    for i, chunk in enumerate(context_chunks):
                        source_info = os.path.basename(sources[i % len(sources)] if sources else "Unknown")
                        st.markdown(f"**Chunk {i + 1}** (from `{source_info}`):")
                        st.code(chunk, language='markdown')

            st.markdown("---")
