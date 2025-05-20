# test_faiss_content.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

results = retriever.get_relevant_documents("What is sustainability?")
print(f"Retrieved {len(results)} documents.")
for doc in results:
    print(doc.page_content[:200])
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 80)
# Check if the retrieved documents are not empty
assert len(results) > 0, "No documents retrieved."
# Check if the retrieved documents contain relevant content
for doc in results:
    assert "sustainability" in doc.page_content.lower(), "Retrieved document does not contain relevant content."
# Check if the source metadata is present
for doc in results:
    assert "source" in doc.metadata, "Source metadata is missing."
# Check if the source metadata is not empty
for doc in results:
    assert doc.metadata["source"], "Source metadata is empty."