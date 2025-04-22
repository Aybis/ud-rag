import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()
def stream_response(chunks):
    for chunk in chunks:
        yield chunk.content

# Set the app name
st.set_page_config(page_title="UD Kaizen")

# ======================{Sidebar Menu}=====================
# Sidebar menu
st.sidebar.title("Menu")
selected_model = st.sidebar.selectbox(
    "LLM Model:",
    ["Gemini", "ChatGPT"],
    index=0,
    help="Select which Large Language Model to use for answering your questions."
)
menu_option = st.sidebar.radio("Select an option:", ["Chat", "Base Knowledge", "About"])

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
memory = st.session_state.memory

if menu_option == "Chat":
    st.title("UD Kaizen Chatbot")
    st.markdown("### Select Knowledge Base Model")
    selected_knowledge_base = st.selectbox("Knowledge Base:", ["Gemini", "OpenAI"])
    use_knowledge_base = st.toggle("Use Knowledge Base (RAG)", value=True)
    st.write("Welcome to the UD Kaizen Chatbot! Ask me anything.")
    google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")  # Moved here
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "history" not in st.session_state:
        st.session_state.history = ChatMessageHistory()

    # Display previous messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; align-items: center; margin-bottom: 0.5rem;'>
                    <div style='background-color: #2B2B2B; padding: 10px 16px; border-radius: 16px; max-width: 70%; color: white; text-align: right;'>
                        {message["content"]}
                    </div>
                    <div style='margin-left: 0.5rem; font-size: 24px;'>🧑‍💻</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("bot", avatar="🤖"):
                st.markdown(f"<div>{message['content']}</div>", unsafe_allow_html=True)

    # Input box at bottom
    prompt = st.chat_input("Type your message...")

    if prompt:
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; align-items: center; margin-bottom: 0.5rem;'>
                    <div style='background-color: #2B2B2B; padding: 10px 16px; border-radius: 16px; max-width: 70%; color: white; text-align: right;'>
                        {prompt}
                    </div>
                    <div style='margin-left: 0.5rem; font-size: 24px;'>🧑‍💻</div>
                </div>
                """, unsafe_allow_html=True)

        # Load key from env
        openai_api_key = os.getenv("OPEN_AI_KEY")

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        if not use_knowledge_base:
            # General chat mode
            if selected_model == "ChatGPT":
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    streaming=True,
                    openai_api_key=openai_api_key,
                    callback_manager=callback_manager
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.7,
                    streaming=True,
                    google_api_key=google_api_key,
                    callback_manager=callback_manager
                )

            with st.chat_message("bot", avatar="🤖"):
                response_container = st.empty()
                user_message = HumanMessage(content=prompt)
                response = ""
                for chunk in llm.stream([user_message]):
                    if hasattr(chunk, "content"):
                        response += chunk.content
                        response_container.markdown(response + "▌")
                response_container.markdown(response)
                st.session_state.messages.append({"role": "bot", "content": response})
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(response)
            st.stop()

        if use_knowledge_base:
            # Set up embedding model
            if selected_knowledge_base == "OpenAI":
                embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
            else:
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

            # Load FAISS indexes from knowledge_base
            base_path = os.path.join("knowledge_base", selected_knowledge_base.lower())
            retrievers = []
            if os.path.exists(base_path):
                for folder in os.listdir(base_path):
                    store_path = os.path.join(base_path, folder)
                    if os.path.isdir(store_path):
                        try:
                            vs = FAISS.load_local(store_path, embeddings=embedding_model, allow_dangerous_deserialization=True, index_name="index")
                            retrievers.append(vs.as_retriever())
                        except Exception as e:
                            st.warning(f"Could not load vector store from {folder}: {e}")
            if retrievers:
                all_docs = []
                for r in retrievers:
                    try:
                        docs = r.vectorstore.similarity_search("a", k=5)
                        all_docs.extend(docs)
                    except Exception as e:
                        st.warning(f"Could not extract documents: {e}")

                if all_docs:
                    combined_vs = FAISS.from_documents(all_docs, embedding_model)
                    retriever = combined_vs.as_retriever()
                else:
                    st.error("⚠️ No documents could be retrieved from the knowledge base.")
                    st.stop()
            else:
                st.error("⚠️ No knowledge base found. Please upload files in the Base Knowledge section first.")
                st.stop()

            # Set up streaming LLM
            if selected_model == "ChatGPT":
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    streaming=True,
                    openai_api_key=openai_api_key,
                    callback_manager=callback_manager
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.7,
                    streaming=True,
                    google_api_key=google_api_key,
                    callback_manager=callback_manager
                )

            # Update prompt with context
            system_prompt = """
            Anda adalah asisten cerdas bernama 'UD Kaizen Helper' yang ahli dalam menjawab pertanyaan terkait produk-produk UD Trucks.
            Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
            Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.
            
            - Jika pengguna bertanya tentang list produk, cari dan tampilkan semua nama produk dari seluruh dokumen yang tersedia. Jangan hanya berdasarkan pada satu bagian. Jika ada banyak, buat daftarnya selengkap mungkin. Sertakan semuanya dalam bentuk list bernomor.
            - Jika pengguna bertanya tentang spesifikasi produk:
            - Berikan penjelasan yang lengkap dan detail.
            - Tambahkan opini atau analisis dari asisten terkait keunggulan atau kekurangan produk tersebut, serta potensi penggunaannya.
            - Ajukan pertanyaan lanjutan untuk membantu pengguna mengeksplorasi produk lebih dalam.

            - Jika pengguna meminta perbandingan atau comparison antar produk:
            - Tampilkan informasi dalam bentuk tabel markdown yang rapi dan ringkas.
            - Sertakan ringkasan setelah tabel.
            - Tambahkan opini atau kesimpulan dari asisten tentang produk mana yang lebih unggul dan mengapa. Berikan analisis berdasarkan konteks dan informasi yang tersedia.
            - Ajukan pertanyaan lanjutan untuk memperdalam diskusi atau membantu pengguna memilih produk terbaik sesuai kebutuhan.

            {context}
            """


            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("Riwayat: {chat_history}\nPertanyaan: {question}")
            ])

            chat_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                return_source_documents=True,
                output_key="answer"
            )

            # Stream and display bot response
            with st.chat_message("bot", avatar="🤖"):
                response_container = st.empty()
                relevant_docs = retriever.get_relevant_documents(prompt)
                context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
                formatted_prompt = prompt_template.format_messages(
                    chat_history=memory.chat_memory.messages,
                    question=prompt,
                    context=context_text
                )

                response = ""
                for chunk in llm.stream(formatted_prompt):
                    if hasattr(chunk, "content"):
                        response += chunk.content
                        response_container.markdown(response + "▌")

                response_container.markdown(response)
                # Show source files after the response
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))
                if sources:
                    with st.expander("📄 View Source Files"):
                        for i, src in enumerate(sources, 1):
                            file_url = f"/{selected_knowledge_base.lower()}/{src}"
                            st.markdown(f"{i}. [📄 {src}]({file_url})", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "bot", "content": response})
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(response)

elif menu_option == "Base Knowledge":
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import shutil

    st.title("Base Knowledge")
    st.write("Upload and embed files to use as knowledge base for the chatbot.")

    # Choose embedding model
    embed_model_name = st.selectbox("Choose embedding model", ["Gemini (embedding-001)", "OpenAI (text-embedding-ada-002)"])
    openai_api_key = os.getenv("OPEN_AI_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if embed_model_name.startswith("OpenAI"):
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    else:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    # Upload files
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt", "docx"])
    base_path = os.path.join("knowledge_base", "openai" if embed_model_name.startswith("OpenAI") else "gemini")

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(base_path, file.name)
            os.makedirs(file_path, exist_ok=True)
            full_raw_path = os.path.join(file_path, file.name)
            with open(full_raw_path, "wb") as f:
                f.write(file.read())

            # Load document
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(full_raw_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(full_raw_path)
            else:
                loader = TextLoader(full_raw_path)

            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = splitter.split_documents(documents)

            # Embed and save to FAISS
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            vectorstore.save_local(file_path)

        st.success("Files successfully embedded and saved.")

    # Show embedded file list
    st.subheader("📂 Knowledge Base Files (Grouped by Model)")
    for model_name in ["openai", "gemini"]:
        model_path = os.path.join("knowledge_base", model_name)
        if os.path.exists(model_path):
            st.markdown(f"### 📁 {model_name.upper()} Knowledge Base")
            for folder in os.listdir(model_path):
                file_folder = os.path.join(model_path, folder)
                if os.path.isdir(file_folder):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"✅ {folder}")
                    with col2:
                        if st.button(f"Delete", key=f"del_{model_name}_{folder}"):
                            shutil.rmtree(file_folder)
                            st.rerun()

elif menu_option == "About":
    # ======================{About Feature}=====================
    st.title("About")
    st.write("This is a simple chatbot application built with Streamlit.")