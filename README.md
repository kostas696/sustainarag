# SustainaRAG: Sustainable Business Assistant with RAG

**SustainaRAG** is a lightweight Retrieval-Augmented Generation (RAG) assistant that helps SMEs explore and develop sustainable business strategies. It leverages LangChain, FAISS for retrieval, and the HuggingFace-hosted `zephyr-7b-beta` model for generation. The assistant enables question-answering on ESG/CSR topics from a curated knowledge base of sustainability reports.

---

## 🌍 Problem Statement

Small and medium-sized enterprises (SMEs) often struggle to navigate sustainability frameworks such as the GRI Standards or the SDGs. SustainaRAG provides accessible, AI-assisted guidance by combining semantic search with large language models to answer sustainability questions grounded in real documents.

---

## 🎯 Project Objectives

- Ingest PDF documents into a FAISS vector store using LangChain.
- Retrieve relevant chunks via similarity search.
- Use HuggingFace's `zephyr-7b-beta` LLM for response generation.
- Deliver results via an interactive Streamlit app.

---

## 📁 Project Structure

```
.
├── LICENSE                       # MIT license
├── README.md                    # This file
├── app.py                       # Streamlit frontend
├── ingest.py                    # PDF ingestion & FAISS index creation
├── retriever.py                 # LangChain QA pipeline
├── requirements.txt             # Dependencies
├── data/                        # Sustainability PDFs
├── faiss_index/                 # FAISS index files (.faiss and .pkl)
├── assets/                      # Reserved for optional visual assets
└── tests/                       # Unit tests for retriever and LLM
    ├── test_faiss_content.py
    └── test_llm_call.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/sustainarag.git
cd sustainarag
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> ✅ Add your Hugging Face API key to a `.env` file:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## 🚀 Run the Assistant

```bash
streamlit run app.py
```

> ℹ️ If you encounter errors due to `torch.classes`, make sure this config exists:

**`.streamlit/config.toml`**
```toml
[server]
fileWatcherType = "none"
```

This disables Streamlit’s file-watcher that conflicts with `torch`.

---

## 💬 Sample Queries

- What are the environmental disclosure requirements under GRI?
- How can SMEs support the Sustainable Development Goals?
- Give examples of corporate responsibility best practices.

---

## 🧪 Testing

```bash
pytest tests/
```

Covers:
- `test_faiss_content.py`: Ensures proper document ingestion and retrieval
- `test_llm_call.py`: Verifies HuggingFace LLM call returns valid output

---

## 🧰 Technologies Used

- Python 3.12
- Streamlit
- LangChain
- FAISS
- Hugging Face Hub (`HuggingFaceH4/zephyr-7b-beta`)
- PyPDFLoader
- python-dotenv, tqdm, pytest

---

## 📈 Future Enhancements

- Add support for more documents and industries
- Integrate document summarization per query
- Switchable model endpoints (OpenRouter, DeepSeek, etc.)
- Add citations and source highlighting in the output
- Implement persistent memory and session analytics

---

🧾 Author: Konstantinos Soufleros
📅 Date: May 2025

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).