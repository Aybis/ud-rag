import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
import os
import tempfile
import shutil

# Constants
VECTORSTORE_DIR = "vectorstore/faiss_index"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Sidebar - UI navigation
st.sidebar.title("📁 Document RAG")
menu = st.sidebar.radio("Menu", ["New Chat", "Uploaded Files"])

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# File upload and embedding
if menu == "New Chat":
    st.title("🧠 Ask Questions From PDF Documents")

    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        all_docs = []

        for uploaded_file in uploaded_files:
            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Track uploaded files
            st.session_state.uploaded_files.append(uploaded_file.name)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            all_docs.extend(chunks)

        if all_docs:
            db = FAISS.from_documents(all_docs, embeddings)
            db.save_local(VECTORSTORE_DIR)
            st.success("Embeddings generated and FAISS index saved!")

elif menu == "Uploaded Files":
    st.title("📄 Uploaded & Embedded Files")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.markdown(f"- {f}")
    else:
        st.info("No files uploaded yet.")

# Main chat interface
index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
if os.path.exists(index_path):
    db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", k=3)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
            st.success(result)
else:
    st.warning("⚠️ No FAISS index found. Please upload PDF(s) first from the 'New Chat' menu.")