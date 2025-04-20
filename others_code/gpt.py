import streamlit as st
import os
import tempfile
import glob
import pickle
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

# Set constants for file paths
DOCS_FOLDER = "docs"
EMBEDDINGS_FOLDER = "embeddings"
VECTOR_STORE_PATH = os.path.join(EMBEDDINGS_FOLDER, "faiss_index")

# Create necessary directories if they don't exist
os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="UD Kaizen Helper", page_icon="🚚", layout="wide")

# Configure page to properly display markdown
st.markdown("""
<style>
    .element-container .stMarkdown p {
        white-space: pre-wrap;
    }
    .element-container table {
        width: 100%;
    }
    .element-container th, .element-container td {
        text-align: left;
        padding: 8px;
    }
    .element-container tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .element-container th {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to get list of files in the docs folder
def get_pdf_files_in_docs():
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    return [os.path.basename(file) for file in pdf_files]

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to create text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Function to save vector store to disk
def save_vector_store(vector_store):
    vector_store.save_local(VECTOR_STORE_PATH)
    
# Function to load vector store from disk if it exists
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    return None

# Callback handler for streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌", unsafe_allow_html=True)

# Function to create conversation chain
def get_conversation_chain(vector_store):
    # Prompt to support markdown formatting
    system_prompt = """
    Anda adalah asisten cerdas bernama 'UD Kaizen Helper' yang ahli dalam menjawab pertanyaan terkait produk-produk UD Trucks.
    Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
    Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.

    - Jika pengguna bertanya tentang list produk, cari dan tampilkan semua nama produk dari seluruh dokumen yang tersedia. Jangan hanya berdasarkan pada satu bagian. Jika ada banyak, buat daftarnya selengkap mungkin. Sertakan semuanya dalam bentuk list bernomor.
    - Jika pengguna bertanya tentang spesifikasi produk, berikan penjelasan yang lengkap dan detail.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan informasi dalam bentuk tabel markdown yang rapi dan ringkas serta sertakan ringkasan.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan summary pendapatmu tentang produk tersebut.

    {context}
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=system_prompt + "\n\nRiwayat Percakapan:\n{chat_history}\n\nPertanyaan: {question}\nJawaban:"
    )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7,
        streaming=True
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    # Save a reference to the LLM in the session state so it can be accessed later
    st.session_state.llm = llm
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# App title and description
st.title("🚚 UD Kaizen Helper")
st.markdown("""
Upload UD Trucks brosur PDF dan ajukan pertanyaan tentang produk-produk UD Trucks.
""")

# Check if API key exists in environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found in environment variables. Please add it to your .env file.")
    st.stop()

# Check if a vector store exists and load it
existing_vector_store = load_vector_store()
if existing_vector_store and "conversation" not in st.session_state:
    st.session_state.conversation = get_conversation_chain(existing_vector_store)
    st.session_state.pdf_processed = True
    st.success("Loaded existing document embeddings.")

# Get list of PDFs in docs folder
docs_pdfs = get_pdf_files_in_docs()

# Sidebar for file uploads and processing options
with st.sidebar:
    st.subheader("Document Sources")
    
    source_option = st.radio(
        "Choose document source:",
        ["Upload Files", "Use Files from 'docs' Folder"]
    )
    
    if source_option == "Upload Files":
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process Uploaded Files"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing your PDFs..."):
                    # Save the uploaded PDFs to the docs folder
                    for pdf in pdf_docs:
                        pdf_path = os.path.join(DOCS_FOLDER, pdf.name)
                        with open(pdf_path, "wb") as f:
                            f.write(pdf.getvalue())
                    
                    # Get all PDFs from the docs folder
                    all_pdfs = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
                    
                    # Extract text from PDFs
                    raw_text = extract_text_from_pdf(all_pdfs)
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks")
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    st.write("Created vector store with embeddings")
                    
                    # Save vector store
                    save_vector_store(vector_store)
                    st.write("Saved vector store to disk")
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.pdf_processed = True
                    
                    st.success("Processing complete! You can now chat with your documents.")
    else:
        if docs_pdfs:
            st.write(f"Found {len(docs_pdfs)} PDF files in 'docs' folder:")
            selected_pdfs = st.multiselect(
                "Select PDFs to process:",
                docs_pdfs,
                default=docs_pdfs
            )
            
            if st.button("Process Selected Files"):
                if not selected_pdfs:
                    st.error("Please select at least one PDF file!")
                else:
                    with st.spinner("Processing selected PDFs..."):
                        # Get full paths for selected PDFs
                        selected_pdf_paths = [os.path.join(DOCS_FOLDER, pdf) for pdf in selected_pdfs]
                        
                        # Extract text from PDFs
                        raw_text = extract_text_from_pdf(selected_pdf_paths)
                        
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.write(f"Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        vector_store = get_vector_store(text_chunks)
                        st.write("Created vector store with embeddings")
                        
                        # Save vector store
                        save_vector_store(vector_store)
                        st.write("Saved vector store to disk")
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.session_state.pdf_processed = True
                        
                        st.success("Processing complete! You can now chat with your documents.")
        else:
            st.warning("No PDF files found in the 'docs' folder. Please add some PDFs to the folder or upload files.")
            
    st.subheader("Vector Store Management")
    if st.button("Clear Vector Store"):
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
            st.success("Vector store cleared successfully.")
            st.session_state.pdf_processed = False
            if "conversation" in st.session_state:
                del st.session_state.conversation
            st.experimental_rerun()
            with st.spinner("Processing your PDFs..."):
                # Save the uploaded PDFs to a temporary directory
                temp_dir = tempfile.mkdtemp()
                temp_pdf_paths = []
                
                for pdf in pdf_docs:
                    temp_pdf_path = os.path.join(temp_dir, pdf.name)
                    with open(temp_pdf_path, "wb") as f:
                        f.write(pdf.getvalue())
                    temp_pdf_paths.append(temp_pdf_path)
                
                # Extract text from PDFs
                raw_text = extract_text_from_pdf(temp_pdf_paths)
                
                # Create text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(f"Created {len(text_chunks)} text chunks")
                
                # Create vector store
                vector_store = get_vector_store(text_chunks)
                st.write("Created vector store with embeddings")
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.session_state.pdf_processed = True
                
                # Clean up temporary files
                for path in temp_pdf_paths:
                    os.remove(path)
                os.rmdir(temp_dir)
                
                st.success("Processing complete! You can now chat with your documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize PDF processed flag
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask something about your documents...")

# Process user query
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not st.session_state.pdf_processed:
            response = "Please upload and process at least one PDF document first."
            st.markdown(response)
        else:
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Set up the streaming handler
                stream_handler = StreamHandler(message_placeholder)
                
                # Get the LLM from session state and update it with streaming handler
                streaming_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    streaming=True,
                    callbacks=[stream_handler]
                )
                
                # Create a copy of the conversation with the streaming LLM
                retriever = st.session_state.conversation.retriever
                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    output_key='answer'
                )
                
                # Re-define prompt
                prompt = PromptTemplate(
                    input_variables=["context", "question", "chat_history"],
                    template="""
                    Anda adalah asisten cerdas bernama 'UD Kaizen Helper' yang ahli dalam menjawab pertanyaan terkait produk-produk UD Trucks.
                    Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
                    Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.
                    
                    - Jika pengguna bertanya tentang list produk, cari dan tampilkan semua nama produk dari seluruh dokumen yang tersedia. Jangan hanya berdasarkan pada satu bagian. Jika ada banyak, buat daftarnya selengkap mungkin. Sertakan semuanya dalam bentuk list bernomor.
                    - Jika pengguna bertanya tentang spesifikasi produk, berikan penjelasan yang lengkap dan detail.
                    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan informasi dalam bentuk tabel markdown yang rapi dan ringkas serta sertakan ringkasan.
                    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan summary pendapatmu tentang produk tersebut.
                    
                    {context}
                    
                    Riwayat Percakapan:
                    {chat_history}
                    
                    Pertanyaan: {question}
                    Jawaban:
                    """
                )
                
                # Create new streaming chain
                streaming_chain = ConversationalRetrievalChain.from_llm(
                    llm=streaming_llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": prompt},
                    return_source_documents=True,
                    output_key="answer"
                )
                
                # Process the query
                result = streaming_chain({"question": user_query})
                response = result["answer"]
                
                # Display the final response with markdown formatting
                message_placeholder.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                response = f"Error: {str(e)}. Please check your connection and try again."
                message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and OpenAI.")