import os
import io
import logging
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader

# Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API
load_dotenv()
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

DOCS_FOLDER = "docs/"
INDEX_PATH = "faiss_index"

#UI
st.set_page_config(page_title="IsuzuBOT", page_icon="🚗")

# --- PDF PROCESSING  ---
def get_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            try:
                pdf_reader = PdfReader(filepath)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                if text:
                    documents.append({"text": text, "source": filename, "filepath": filepath})
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    return documents

#split to chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

#vector store 
def create_vector_store(documents):
    texts = []
    metadatas = []
    for doc in documents:
        chunks = get_text_chunks(doc["text"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"], "filepath": doc["filepath"]}] * len(chunks))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(INDEX_PATH)
    return vector_store

@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner("Processing documents..."):
            documents = get_documents()
            vector_store = create_vector_store(documents)
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

vector_store = load_faiss_index()

# system prompt
def get_conversational_chain():
    retriever = vector_store.as_retriever()

    # Prompt to support markdown formatting
    system_prompt = """
    Anda adalah asisten chatbot bernama 'UD-BOT' yang ahli dalam menjawab pertanyaan terkait produk-produk UD Trucks.
    Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
    Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.

    - Jika pengguna bertanya tentang spesifikasi produk, berikan penjelasan yang lengkap dan detail.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan informasi dalam bentuk tabel markdown yang rapi dan ringkas serta sertakan ringkasan.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan summary pendapatmu tentang produk tersebut.

    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_GEMINI_API_KEY, temperature=0.9)

    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    chain = (
        RunnableMap({
            "context": lambda x: retriever.get_relevant_documents(x["question"]),
            "input": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }) | document_chain
    )

    return chain

#get answers
def user_input(user_question):
    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "Sorry, I couldn't find the answer.", None
    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        path = doc.metadata.get("filepath", None)
        if src not in seen:
            seen.add(src)
            sources.append((src, path))
    chat_history = [
        (HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        for msg in st.session_state.messages if msg["role"] in ("user", "assistant")
    ]
    
    chain = get_conversational_chain()
    response = chain.invoke({
        "question": user_question,
        "chat_history": chat_history
    })
    
    answer_text = response if isinstance(response, str) else getattr(response, "content", str(response))
    return answer_text, sources

# --- STREAMLIT UI ---
st.title("UD-BOT 🚗")
with st.sidebar:
    st.title("Knowledge Base")
    if os.path.exists(DOCS_FOLDER):
        files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
        if files:
            for i, file in enumerate(sorted(files), 1):
                st.markdown(f"{i}. {file}")
        else:
            st.write("No PDF files found.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Tanyakan apapun mengenai UD Trucks"}]
#menyimpan referensi dokumen dari jawaban
if "last_reference" not in st.session_state:
    st.session_state.last_reference = None

#chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown("""
            <div style='display: flex; justify-content: flex-end; align-items: center; gap: 8px;'>
                <div style='background-color: #1E1E1E; padding: 10px 16px; border-radius: 16px; max-width: 75%; text-align: right;'>
                    <span style='color: white;'>{}</span>
                </div>
                <div style='font-size: 24px;'>🧑‍💼</div>
            </div>
            """.format(message["content"]), unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask about UD Trucks products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown("""
            <div style='display: flex; justify-content: flex-end; align-items: center; gap: 8px;'>
                <div style='background-color: #1E1E1E; padding: 10px 16px; border-radius: 16px; max-width: 75%; text-align: right;'>
                    <span style='color: white;'>{}</span>
                </div>
                <div style='font-size: 24px;'>🧑‍💼</div>
            </div>
            """.format(prompt), unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_text, source_info = user_input(prompt)
                st.write(answer_text)
                # Simpan referensi dokumen ke session state
                st.session_state.last_reference = source_info
        
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
    
    if st.session_state.last_reference:
        st.markdown("Sources:")
        for source, filepath in st.session_state.last_reference:
            if filepath and os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    file_bytes = f.read()
                st.download_button(
                    label=f"Download {source}",
                    data=file_bytes,
                    file_name=source,
                    mime="application/pdf"
                )