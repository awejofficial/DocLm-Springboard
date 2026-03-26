from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# -------------------------------
# Initialize LLM
# -------------------------------
llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.7
)

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------
vectorstore = None
retriever = None
status_callback = None  # UI status hook

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENT_PATH = os.path.join(BASE_DIR, "documents")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def set_status_callback(fn):
    global status_callback
    status_callback = fn


def _status(msg: str):
    global status_callback
    print(msg)
    if status_callback:
        status_callback(msg)


def build_vectorstore():
    global retriever, vectorstore

    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    _status("📂 Scanning documents folder...")

    pdf_loader = DirectoryLoader(
        path=DOCUMENT_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    txt_loader = DirectoryLoader(
        path=DOCUMENT_PATH,
        glob="*.txt",
        loader_cls=TextLoader
    )

    _status("📄 Loading PDF documents...")
    pdf_docs = list(pdf_loader.lazy_load())

    _status("📝 Loading TXT documents...")
    txt_docs = list(txt_loader.lazy_load())

    docs = pdf_docs + txt_docs
    _status(f"✅ Found {len(pdf_docs)} PDF(s) and {len(txt_docs)} TXT file(s)")

    if not docs:
        _status("⚠️ No documents found. Please upload files first.")
        retriever = None
        return

    _status("✂️ Splitting documents into chunks (size=300, overlap=30)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    splitted_docs = splitter.split_documents(docs)
    _status(f"📦 Created {len(splitted_docs)} text chunks")

    _status("🧠 Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    _status("✅ Embedding model loaded")

    _status("🗄️ Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )

    _status("🚀 Vector DB ready! You can now ask questions.")


def format_docs(docs):
    context = "\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
    return context, sources


def corrective_rag(query):
    global retriever

    if retriever is None:
        return "No documents available. Please upload files first.", []

    _status("🔍 Retrieving relevant chunks from vector store...")
    retrieved_docs = retriever.invoke(query)
    context, sources = format_docs(retrieved_docs)

    _status("🤔 Checking relevance of retrieved documents...")
    evaluation_prompt = f"""
    Query: {query}
    Retrieved Context:
    {context}
    Are these documents relevant enough to answer the query?
    Respond strictly with YES or NO.
    """
    evaluation = llm.invoke(evaluation_prompt).content.strip()
    _status(f"📊 Relevance check result: {evaluation}")

    if "NO" in evaluation.upper():
        _status("✏️ Rewriting query for better retrieval...")
        rewrite_prompt = f"""
        The query '{query}' did not retrieve relevant documents.
        Rewrite it to improve retrieval quality.
        Only return the improved query.
        """
        improved_query = llm.invoke(rewrite_prompt).content.strip()
        _status(f"🔄 Improved query: {improved_query[:60]}...")
        retrieved_docs = retriever.invoke(improved_query)
        context, sources = format_docs(retrieved_docs)

    _status("💬 Generating final answer with LLaMA 3.1...")
    final_prompt = f"""
    Answer the question using ONLY the context below.
    Context:
    {context}
    Question: {query}
    Also mention the sources used at the end.
    """
    answer = llm.invoke(final_prompt)
    _status("✅ Answer ready!")
    return answer.content, sources


def get_answer(query: str):
    global retriever, vectorstore
    if retriever is None:
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            _status("📂 Loading existing vector store from disk...")
            from langchain_community.vectorstores import Chroma
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_model,
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "lambda_mult": 0.7}
            )
            _status("✅ Vector store loaded from disk!")
        else:
            _status("🔄 No vector store found. Building now...")
            build_vectorstore()
    if retriever is None:
        return "No documents found. Please upload documents first.", []
    answer, sources = corrective_rag(query)
    return answer, sources
