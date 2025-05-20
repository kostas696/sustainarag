import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_documents(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}")
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(documents)

def embed_documents(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)
    print(f"FAISS index saved to {path}")

if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    vs = embed_documents(chunks)
    save_vectorstore(vs, INDEX_DIR)
