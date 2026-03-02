📚 DocLM – Groq Powered RAG Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

DocLM is a Retrieval-Augmented Generation (RAG) system that enables intelligent document querying using semantic search and Groq-hosted LLMs.

It allows users to upload documents (PDF/TXT), build a vector database, and ask natural language questions while receiving context-aware answers with source citations.

🚀 Key Features

📄 PDF & TXT document ingestion

✂️ Intelligent text chunking

🧠 Embedding generation for semantic search

🗂 Chroma vector database

🔎 Configurable Top-K retrieval

🤖 Groq LLM integration (Llama 3.1-8B)

📝 Context-aware answer synthesis

📖 Source citation display

🎨 Modern Streamlit chat-style UI

🧠 System Architecture

Document Loader

Metadata Cleaning

Text Splitter

Embedding Model

Chroma Vector Store

Retriever (Top-K Similarity Search)

Groq LLM (Answer Generation)

Streamlit Interface

⚙️ Tech Stack

Python 3.10+

LangChain

ChromaDB

Groq API

Streamlit

Sentence Transformers

🔧 Installation
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install langchain-chroma langchain-groq
uv run streamlit run app.py
🔐 Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
📈 Future Improvements

Conversational memory support

Persistent vector database storage

Hybrid search (BM25 + Embeddings)

Local LLM fallback

Multi-document indexing optimization

🎯 Project Goal

This project was developed to understand and implement end-to-end Retrieval-Augmented Generation systems, including vector databases, retrieval tuning, and LLM-powered answer synthesis.
