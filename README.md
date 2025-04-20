# 🧠 RAG App with Streamlit, FAISS, LangChain & OpenAI Embeddings

This app allows you to upload multiple PDF documents, converts them into vector embeddings using OpenAI's `text-embedding-ada-002` model, and enables you to ask questions against those documents using Retrieval-Augmented Generation (RAG).

## 📦 Features

- Upload multiple PDFs
- Embeds content using `text-embedding-ada-002`
- Stores embeddings in FAISS vector index
- Queries documents using LangChain with Mistral LLM
- Simple Streamlit web interface
- Sidebar with chat and file list

## 🛠️ Installation

1. Clone the repo and navigate to the project directory:

   ```bash
   git clone https://github.com/your-username/ud-rag.git
   cd ud-rag
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ```

## 🚀 Usage

Run the app using Streamlit:

```bash
streamlit run main.py
```

- Upload PDF files from the main screen
- Ask questions about the content
- Switch views using the sidebar

## 📁 Project Structure

```
ud-rag/
├── main.py             # Streamlit app
├── ingest.py           # (Optional) batch loader if needed
├── vectorstore/        # FAISS index storage
├── data/               # PDF files
├── .env                # Environment variables
└── requirements.txt    # Dependencies
```

## 📚 Built With

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)

## 📝 License

MIT License
