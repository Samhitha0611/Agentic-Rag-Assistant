# Agentic RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to ask questions about documents stored in a knowledge base.

The application retrieves relevant document chunks using semantic search, builds context from the retrieved information, and uses Google's Gemini model to generate answers based only on the provided documents.

## Features

- Ask questions about uploaded documents
- Retrieve relevant document chunks using semantic search
- Generate embeddings using Sentence Transformers
- Store and search document embeddings using ChromaDB
- Build contextual prompts from retrieved documents
- Generate AI-powered answers using Google Gemini
- Display source documents used to generate answers
- Avoid answering questions when relevant information is not available
- Suggested questions for quick interaction
- Chat history using Streamlit session state
- Clear chat functionality

## Tech Stack

- Python
- Streamlit
- LangChain
- Sentence Transformers
- Hugging Face
- ChromaDB
- Google Gemini API
- PyPDF

## Project Structure

```text
Agentic-RAG-Assistant/
│
├── app/
│
├── data/
│   ├── chroma_db/              # Generated ChromaDB vector database
│   └── documents/              # Documents for the knowledge base
│
├── docs/
│
├── notebooks/
│
├── scripts/
│   ├── __init__.py
│   ├── context_builder.py      # Builds context from retrieved documents
│   ├── ingest.py               # Loads and processes documents
│   ├── llm.py                  # Gemini configuration
│   ├── rag_pipeline.py         # Complete RAG pipeline
│   └── retrieval.py            # Retrieves relevant documents
│
├── tests/                      # Testing files
│
├── .env                        # Environment variables
├── .gitignore
├── main.py                     # Streamlit application
├── README.md
└── requirements.txt