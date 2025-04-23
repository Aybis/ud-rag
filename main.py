import os
import requests
from bs4 import BeautifulSoup
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
st.sidebar.title("Menu")
selected_model = st.sidebar.selectbox(
    "LLM Model:",
    ["Gemini", "ChatGPT"],
    index=0,
    help="Select which Large Language Model to use for answering your questions."
)
selected_knowledge_base = st.sidebar.selectbox("Knowledge Base", ["Gemini", "OpenAI"])
st.session_state.selected_knowledge_base = selected_knowledge_base

menu_option = st.sidebar.radio("Select an option:", ["Chat", "Base Knowledge", "Inspect FAISS File", "About"])
st.sidebar.markdown("## ⚙️ Settings")
use_knowledge_base = st.sidebar.toggle("Use Knowledge Base (RAG)", value=True)
enable_web_scraping = st.sidebar.toggle("Enable Web Scraping (Google Search)", value=False)


# Display full diagnostic of LLM + Embedding + RAG
selected_knowledge_base_diag = st.session_state.get("selected_knowledge_base", None)
st.sidebar.markdown("### 📊 Current Configuration")
st.sidebar.write("🧠 **LLM Foundation Model + Prompt Embedding**")
st.sidebar.code(f"{selected_model}", language="yaml")

st.sidebar.write("📚 **Knowledge Base Embedding Source** (RAG)")
if selected_knowledge_base_diag:
    st.sidebar.code(f"{selected_knowledge_base_diag}", language="yaml")
else:
    st.sidebar.write("_Not selected yet_")

st.sidebar.info("🔬 You're testing combinations like:\n\nLLM + Prompt Embedding × RAG Source")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
memory = st.session_state.memory

if menu_option == "Chat":
    st.title("UD Kaizen Chatbot")
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
                if "sources" in message and message["sources"]:
                    if all(not s.startswith("http") for s in message["sources"]):
                        expander_label = "📄 Source Files"
                    else:
                        expander_label = "🌐 Source Websites"
                    with st.expander(expander_label):
                        for i, src in enumerate(message["sources"], 1):
                            if src.startswith("http"):
                                st.markdown(f"{i}. [🔗 {src}]({src})", unsafe_allow_html=True)
                            else:
                                file_name = os.path.basename(src)
                                kb_root = "knowledge_base"
                                model_dirs = os.listdir(kb_root)
                                file_path = None
                                for model_dir in model_dirs:
                                    possible_path = os.path.join(kb_root, model_dir, src)
                                    if os.path.exists(possible_path):
                                        file_path = possible_path
                                        break
                                if file_path and os.path.isfile(file_path):
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"📁 Download {file_name}",
                                            data=f,
                                            file_name=file_name,
                                            mime="application/octet-stream",
                                            key=f"msg_dl_{i}_{file_name}"
                                        )

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

        # ======================{Web Scraping Block for Both Modes}======================
        scraped_docs = []
        scraped_sources = []
        if enable_web_scraping:
            from langchain_core.documents import Document
            serpapi_key = os.getenv("SERP_API_KEY")
            search_url = "https://serpapi.com/search.json"
            params = {
                "q": prompt,
                "api_key": serpapi_key,
                "engine": "google",
                "num": 3
            }
            try:
                st.info("🌐 Web scraping is enabled. Searching...")
                serp_response = requests.get(search_url, params=params)
                search_results = serp_response.json().get("organic_results", [])
                st.write("🔗 Search results:", [r.get("link") for r in search_results])
                for result in search_results[:3]:
                    page_url = result.get("link")
                    page = requests.get(page_url, timeout=10)
                    soup = BeautifulSoup(page.text, "html.parser")
                    content = "\n".join([p.get_text() for p in soup.find_all("p")])
                    if content.strip():
                        doc = Document(page_content=content, metadata={"source": page_url})
                        scraped_docs.append(doc)
                        scraped_sources.append(page_url)
            except Exception as e:
                st.warning(f"Web scraping failed: {e}")

        # ======================{General Chat Mode}======================
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
            # Show LLM SDK and model in use
            llm_provider = type(llm).__name__
            llm_model = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            st.info(f"✅ LLM SDK in use: **{llm_provider}**\n\n🧠 Model: `{llm_model}`")

            # Add web scraping context if available (improved instruction to LLM)
            context_note = ""
            if scraped_docs:
                combined_text = "\n\n".join(doc.page_content for doc in scraped_docs)
                context_note = f"""
Berikut adalah hasil pencarian web dari berbagai sumber untuk membantu menjawab pertanyaan berikut. Gunakan informasi ini sebagai referensi utama jika relevan:

{combined_text}

---
"""
            prompt_with_scraping = context_note + f"Pertanyaan: {prompt}"

            with st.chat_message("bot", avatar="🤖"):
                with st.spinner("🤖 Thinking... Generating response..."):
                    response_container = st.empty()
                    response = ""
                    for chunk in llm.stream([HumanMessage(content=prompt_with_scraping)]):
                        if hasattr(chunk, "content"):
                            response += chunk.content
                            response_container.markdown(response + "▌")
                    response_container.markdown(response)
                # Show scraped sources if available
                if scraped_sources:
                    with st.expander("🌐 Source Websites from Web Scraping"):
                        for i, src in enumerate(scraped_sources, 1):
                            st.markdown(f"{i}. [🔗 {src}]({src})", unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "bot",
                    "content": response,
                    "sources": scraped_sources if not use_knowledge_base else sources
                })
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(response)
            st.stop()

        if use_knowledge_base:
            # Set up embedding model
            if selected_knowledge_base == "OpenAI":
                embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
            else:
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            # Show embedding provider and model
            embedding_provider = type(embedding_model).__name__
            embedding_model_id = getattr(embedding_model, 'model', 'unknown')
            st.info(f"🔎 Embedding Provider: **{embedding_provider}**\n\n📎 Embedding Model: `{embedding_model_id}`")

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
                # Add scraped docs to all_docs if any
                if scraped_docs:
                    all_docs.extend(scraped_docs)
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
            # Show LLM SDK and model in use
            llm_provider = type(llm).__name__
            llm_model = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            st.info(f"✅ LLM SDK in use: **{llm_provider}**\n\n🧠 Model: `{llm_model}`")

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
                relevant_docs = retriever.get_relevant_documents(prompt)
                # Preserve correct source filenames in order
                sources = []
                for doc in relevant_docs:
                    src = doc.metadata.get("source")
                    if src and src not in sources:
                        sources.append(src)
                context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
                formatted_prompt = prompt_template.format_messages(
                    chat_history=memory.chat_memory.messages,
                    question=prompt,
                    context=context_text
                )
                with st.spinner("🤖 Thinking... Generating response..."):
                    response_container = st.empty()
                    response = ""
                    for chunk in llm.stream(formatted_prompt):
                        if hasattr(chunk, "content"):
                            response += chunk.content
                            response_container.markdown(response + "▌")
                    response_container.markdown(response)
                # Always show sources after the response
                if sources:
                    with st.expander("📄 View Source Files"):
                        for i, src in enumerate(sources, 1):
                            if src.startswith("http"):
                                st.markdown(f"""
                                <div style='margin-bottom: 0.5rem;'>
                                    <strong>{i}. 🌐 Source Website:</strong><br>
                                    <a href="{src}" target="_blank" style="text-decoration: none; color: #1E90FF;">{src}</a>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                file_name = os.path.basename(src)
                                file_path = os.path.join("knowledge_base", selected_knowledge_base.lower(), src)
                                if os.path.isfile(file_path):
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"📁 Download {file_name}",
                                            data=f,
                                            file_name=file_name,
                                            mime="application/octet-stream",
                                            key=f"download_{i}"
                                        )
                                else:
                                    continue  # Skip displaying this file if not found
                # Show scraped sources if available (web scraping)
                if scraped_sources:
                    with st.expander("🌐 Source Websites from Web Scraping"):
                        for i, src in enumerate(scraped_sources, 1):
                            st.markdown(f"{i}. [🔗 {src}]({src})", unsafe_allow_html=True)
                # Ensure correct source filenames from RAG are added
                sources = []
                for doc in relevant_docs:
                    src = doc.metadata.get("source")
                    if src and src not in sources:
                        sources.append(src)

                # Combine both knowledge base and web sources
                all_sources = sources + [s for s in scraped_sources if s not in sources]
                st.session_state.messages.append({
                    "role": "bot",
                    "content": response,
                    "sources": all_sources
                })
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
    # Show embedding provider and model
    embedding_provider = type(embedding_model).__name__
    embedding_model_id = getattr(embedding_model, 'model', 'unknown')
    st.info(f"🔎 Embedding Provider: **{embedding_provider}**\n\n📎 Embedding Model: `{embedding_model_id}`")

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
    # Display a subheader for the knowledge base files section
    st.subheader("📂 Knowledge Base Files (Grouped by Model)")

    # Iterate through the two model types: OpenAI and Gemini
    for model_name in ["openai", "gemini"]:
        # Construct the path to the knowledge base directory for the current model
        model_path = os.path.join("knowledge_base", model_name)

        # Check if the directory for the current model exists
        if os.path.exists(model_path):
            # Display the model name as a header
            st.markdown(f"### 📁 {model_name.upper()} Knowledge Base")

            # Iterate through the folders in the model's knowledge base directory
            for folder in os.listdir(model_path):
                file_folder = os.path.join(model_path, folder)

                # Check if the current item is a directory
                if os.path.isdir(file_folder):
                    # Create two columns for displaying the folder name and a delete button
                    col1, col2 = st.columns([4, 1])

                    # Display the folder name in the first column
                    with col1:
                        st.write(f"✅ {folder}")

                    # Add a delete button in the second column
                    with col2:
                        # If the delete button is clicked, remove the folder and refresh the app
                        if st.button(f"Delete", key=f"del_{model_name}_{folder}"):
                            shutil.rmtree(file_folder)  # Delete the folder and its contents
                            st.rerun()  # Refresh the Streamlit app to reflect the changes

elif menu_option == "Inspect FAISS File":
    st.title("🔍 Inspect FAISS Index File")
    st.write("Upload a `.faiss` and `.pkl` file pair to view its document contents.")

    uploaded_faiss = st.file_uploader("Upload .faiss file", type="faiss")
    uploaded_pkl = st.file_uploader("Upload .pkl metadata file", type="pkl")

    if uploaded_faiss and uploaded_pkl:
        import tempfile
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        import shutil

        # Save uploaded files temporarily
        with tempfile.TemporaryDirectory() as tmpdirname:
            faiss_path = os.path.join(tmpdirname, "index.faiss")
            pkl_path = os.path.join(tmpdirname, "index.pkl")

            with open(faiss_path, "wb") as f:
                f.write(uploaded_faiss.read())

            with open(pkl_path, "wb") as f:
                f.write(uploaded_pkl.read())

            try:
                vs = FAISS.load_local(tmpdirname, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                st.success("FAISS index loaded successfully.")

                # Show all documents directly
                all_docs = vs.similarity_search("a", k=100)  # "a" to retrieve everything
                st.markdown("### 📄 All Embedded Documents")
                for i, doc in enumerate(all_docs, 1):
                    st.markdown(f"**{i}. Content Preview:**")
                    st.code(doc.page_content.strip()[:1000])  # limit preview to 1000 chars
                    st.markdown(f"📎 Metadata: `{doc.metadata}`")
                    st.markdown("---")

            except Exception as e:
                st.error(f"Failed to load FAISS index: {e}")

elif menu_option == "About":
    # ======================{About Feature}=====================
    st.title("About")
    st.write("This is a simple chatbot application built with Streamlit.")