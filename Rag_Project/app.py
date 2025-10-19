import streamlit as st
from rag_agentic import AgenticRAG

st.set_page_config(page_title="RAG Chat UI", layout="wide", page_icon="🤖")
st.title("Retrieval-Augmented Generation Chat")

@st.cache_resource(show_spinner=False)
def get_rag_instance():
    return AgenticRAG()

rag = get_rag_instance()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def on_ask():
    user_input = st.session_state.user_input.strip()
    if user_input:
        result = rag.query(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result["answer"]))
        st.session_state.user_input = ""

st.text_input("Ask your question:", key="user_input", on_change=on_ask, placeholder="Type your question here...")

for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
