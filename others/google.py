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

# --- PDF PROCESSING  ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    return splitter.split_text(text)

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

#vector store 
def create_vector_store(documents):
    texts = []
    metadatas = []
    for doc in documents:
        chunks = get_text_chunks(doc["text"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"], "filepath": doc["filepath"]}] * len(chunks))

    if not texts:
        st.warning("No chunks available to embed. Index not updated.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(INDEX_PATH)
    return vector_store

#UI
st.set_page_config(page_title="UD-BOT", page_icon="🚗")

if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "needs_reload" not in st.session_state:
    st.session_state.needs_reload = False

with st.sidebar:
    st.title("Settings")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Saving {uploaded_file.name}..."):
                file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                if uploaded_file.name not in st.session_state.uploaded_file_names:
                    st.session_state.uploaded_file_names.append(uploaded_file.name)
        # Recreate FAISS index with new files
        with st.spinner("Embedding documents..."):
            try:
                documents = get_documents()
                if not documents:
                    st.warning("No valid documents found.")
                else:
                    create_vector_store(documents)
                    st.success("Embedding completed.")
                    st.session_state.needs_reload = True
                    st.rerun()
            except Exception as e:
                st.error(f"⚠️ Re-indexing failed: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    if os.path.exists(DOCS_FOLDER):
        file_options = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
        if file_options:
            st.markdown("### Uploaded Knowledge Base")
            selected_file = st.selectbox("Select file to remove", file_options)
        if st.button("🗑️ Delete Selected File"):
            file_to_delete = os.path.join(DOCS_FOLDER, selected_file)
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                if selected_file in st.session_state.uploaded_file_names:
                    st.session_state.uploaded_file_names.remove(selected_file)
                # Recreate index without the deleted file
                with st.spinner("Updating knowledge base..."):
                    documents = get_documents()
                    if documents:
                        create_vector_store(documents)
                    else:
                        # Clear the FAISS index directory if no documents left
                        import shutil
                        if os.path.exists(INDEX_PATH):
                            shutil.rmtree(INDEX_PATH)
                            os.makedirs(INDEX_PATH)
                        st.warning("All knowledge base files have been removed. Index cleared.")
                    st.session_state.needs_reload = True
                    st.rerun()
    select_model = st.selectbox("Select feature", ["UD-BOT", "Generate Gambar Produk"])

if st.session_state.needs_reload:
    st.cache_resource.clear()
    st.session_state.needs_reload = False

# --- IMAGE SEARCH  ---
def search_google_images(query):
    search_url = "https://www.googleapis.com/customsearch/v1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 1
    }
    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        return results["items"][0]["link"] if "items" in results else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image: {e}")
        return None


@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_file = os.path.join(INDEX_PATH, "index.faiss")

    if os.path.exists(faiss_file):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    documents = get_documents()
    if not documents:
        return None

    with st.spinner("Processing documents..."):
        vector_store = create_vector_store(documents)
        return vector_store if vector_store else None
    
# Load the FAISS index
vector_store = load_faiss_index()

# system prompt
def get_conversational_chain():
    retriever = vector_store.as_retriever()

    # Prompt to support markdown formatting
    system_prompt = """
    Anda adalah asisten chatbot bernama 'UD-BOT' yang ahli dalam menjawab pertanyaan terkait produk-produk UD Trucks.
    Anda boleh menjelaskan kelebihan dan kekurangan produk UD Trucks dibandingkan dengan merk lain.
    Anda boleh menampilkan produk merk lain.
    Walaupun produk UD Truck memiliki kekurangan, And harus unggulkan setiap produk UD Trucks dibandingkan dengan merk lain.
    Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
    Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.

    - Jika pengguna bertanya tentang spesifikasi produk, berikan penjelasan yang lengkap dan detail, lebih mudah dipahami menggunakan tabel markdown yang rapih.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan informasi dalam bentuk tabel markdown yang rapi dan ringkas serta sertakan ringkasan.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan summary pendapatmu tentang produk tersebut.
    - Jika pengguna membandingkan produk dengan merk lain, anda harus mengunggulkan produk UD Trucks dibanding dengan merk lain.

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

if select_model == "Generate Gambar Produk":
    st.subheader("Ketik produk UD yang ingin dicari 🏎️")
    image_prompt = st.text_input("Tipe Produk:")
    if st.button("Search 🚀"):
        with st.spinner("Searching..."):
            image_url = search_google_images(image_prompt)
            if image_url:
                st.image(image_url, caption="", use_container_width=True)
            else:
                st.error("No image found.")

elif select_model == "UD-BOT":
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