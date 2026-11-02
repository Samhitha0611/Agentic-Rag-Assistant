import streamlit as st
from pathlib import Path

from scripts.rag_pipeline import generate_rag_response
from scripts.ingest import ingest_documents

# --------------------------------
# Configuration
# --------------------------------

SHOW_SOURCES = True

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Session state for chat history
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    # Sidebar header
    st.title("🤖 Agentic RAG")
    st.caption("Your AI-powered document assistant")

    # Upload PDF
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    # Process uploaded PDF
    if uploaded_file is not None:

        if st.button("📥 Process Document", use_container_width=True):

            try:
                # -----------------------------------------
                # 1. Create documents directory
                # -----------------------------------------
                documents_dir = Path("data/documents")
                documents_dir.mkdir(parents=True, exist_ok=True)

                # -----------------------------------------
                # 2. Delete old PDF files
                # -----------------------------------------
                for old_file in documents_dir.glob("*.pdf"):
                    old_file.unlink()

                # -----------------------------------------
                # 3. Save new uploaded PDF
                # -----------------------------------------
                file_path = documents_dir / uploaded_file.name

                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                # -----------------------------------------
                # 4. Clear previous chat
                # -----------------------------------------
                st.session_state.messages = []

                # -----------------------------------------
                # 5. Process new document
                # -----------------------------------------
                with st.spinner("Processing document..."):
                    ingest_documents()

                # -----------------------------------------
                # 6. Store current file name
                # -----------------------------------------
                st.session_state.uploaded_file_name = uploaded_file.name

                st.success(
                    f"'{uploaded_file.name}' processed successfully!"
                )

            except Exception as error:
                st.error(
                    f"Error while processing the document: {error}"
                )

    st.divider()

    # About section
    st.subheader("📌 About")

    st.write(
        "Ask questions and get answers based on the documents "
        "stored in your knowledge base."
    )

    st.divider()

    # How it works section
    st.subheader("⚙️ How it works")

    st.markdown("""
    **1. 💬 Ask a question**

    **2. 🔍 Relevant documents are retrieved**

    **3. 🧩 Context is built**

    **4. 🤖 Gemini generates an answer**
    """)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.caption("Powered by RAG • Gemini • ChromaDB")


# ---------------------------------------------------
# Main application
# ---------------------------------------------------

st.title("🤖 Agentic RAG Assistant")

st.caption(
    "Ask questions about the documents in your knowledge base."
)

st.divider()

# ---------------------------------------------------
# Display chat history
# ---------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and SHOW_SOURCES
            and message.get("sources")
        ):
            with st.expander("📚 View Sources"):
                import os

                unique_sources = {
                    os.path.basename(
                        document.metadata.get("source", "Unknown source")
                    )
                    for document, score in message["sources"]
                }

                for source in sorted(unique_sources):
                    st.write(f"📄 {source}")


# Welcome section - shown only when the chat is empty
if not st.session_state.messages:

    st.markdown("### 👋 Welcome to your Agentic RAG Assistant")

    st.write(
        "Ask questions about your uploaded documents and get "
        "AI-powered answers based on your knowledge base."
    )

    st.markdown("### 💡 Try asking:")

    col1, col2 = st.columns(2)

    with col1:
        question_1 = st.button(
            "📄 What is this document about?",
            use_container_width=True
        )

        question_3 = st.button(
            "📑 Summarize the document",
            use_container_width=True
        )

    with col2:
        question_2 = st.button(
            "🔍 What are the key topics?",
            use_container_width=True
        )

        question_4 = st.button(
            "💡 What are the key points?",
            use_container_width=True
        )

    # Check if an example question was clicked
    if question_1:
        st.session_state.suggested_question = (
            "Based on the uploaded document, what is this document about?"
        )
        
    elif question_2:
        st.session_state.suggested_question = (
            "Based on the uploaded document, what are the main topics discussed?"
        )

    elif question_3:
        st.session_state.suggested_question = (
            "Based on the uploaded document, summarize the document."
        )

    elif question_4:
        st.session_state.suggested_question = (
            "Based on the uploaded document, what are the main points discussed?"
        )
# ---------------------------------------------------
# Chat input
# ---------------------------------------------------

user_query = st.chat_input(
    "Ask a question about your documents..."
)

if st.session_state.suggested_question:
    query = st.session_state.suggested_question
    st.session_state.suggested_question = None
else:
    query = user_query

# ---------------------------------------------------
# Process user question
# ---------------------------------------------------

if query:

    # Store and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching the knowledge base..."):
            answer, sources = generate_rag_response(query)

        # Handle system errors
        if answer.startswith("An error occurred"):
            st.error(
                "Sorry, something went wrong while processing your question. "
                "Please try again."
            )

        # Handle no relevant information
        elif answer == "I could not find relevant information in the provided documents.":
            st.warning(answer)

        # Display a normal answer
        else:
            st.markdown(answer)

        # Display sources only when enabled and available
        if SHOW_SOURCES and sources:
            with st.expander("📚 View Sources"):

                import os

                # Get unique source file names
                unique_sources = {
                    os.path.basename(
                        document.metadata.get("source", "Unknown source")
                    )
                    for document, score in sources
                }

                # Display unique sources
                for source in sorted(unique_sources):
                    st.write(f"📄 {source}")

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )