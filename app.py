import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from utils.loaders import load_documents, clean_metadata
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.splitters import split_documents


# LOAD ENV VARIABLES
load_dotenv()

st.set_page_config(
    page_title="📚 RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------- SIDEBAR ---------------------- #
with st.sidebar:
    st.title("⚙️ Settings")

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        st.success("Groq API Key Loaded ✅")
    else:
        st.error("Groq API Key NOT Loaded ❌")

    build_db = st.button("🚀 Build Vector Database")

    st.markdown("---")
    st.markdown("### 🔎 Retrieval Settings")
    k_value = st.slider("Top-K Results", 1, 5, 2)

    st.markdown("---")
    st.caption("Built with ❤️ using LangChain + Groq By Awej")


# ---------------------- HEADER ---------------------- #
st.title("📚 RAG Assistant")
st.markdown("Ask questions based on your uploaded documents.")

# SESSION STATE INIT
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None


# ---------------------- BUILD DATABASE ---------------------- #
if build_db:

    with st.spinner("📂 Loading and processing documents..."):
        documents = load_documents()
        documents = clean_metadata(documents)
        chunks = split_documents(documents)
        embeddings = get_embeddings()
        vectordb = create_vector_store(chunks, embeddings)

        st.session_state.vectordb = vectordb

    st.success("✅ Vector Database Created Successfully!")

    # Show document info
    unique_sources = set(doc.metadata.get("source") for doc in documents)
    st.info(f"Loaded {len(documents)} documents.")
    with st.expander("📁 View Document Sources"):
        for src in unique_sources:
            st.write(f"- {src}")


# ---------------------- QUERY SECTION ---------------------- #
user_query = st.chat_input("Ask a question from your documents...")

if user_query:

    if st.session_state.vectordb is None:
        st.warning("⚠️ Please build the vector database first.")
    else:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("🤖 Generating answer..."):

            retriever = st.session_state.vectordb.as_retriever(
                search_kwargs={"k": k_value}
            )

            docs = retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            prompt = f"""
You are a helpful assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{user_query}

Answer clearly and concisely.
"""

            response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.markdown(response.content)

            # Show sources in expandable format
            if docs:
                with st.expander("📖 View Sources"):
                    sources = list(set(
                        doc.metadata.get("source", "Unknown Source")
                        for doc in docs
                    ))
                    for source in sources:
                        st.write(f"- {source}")