import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

INDEX_DIR = "faiss_index"
HUGGINGFACE_MODEL = "google/flan-t5-base"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_retriever_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a sustainability assistant. Use the provided context to answer the question.
If the context does not contain enough information, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_MODEL,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.3,
        max_new_tokens=256
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return qa_chain