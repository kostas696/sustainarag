import os

os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1" 

from dotenv import load_dotenv
import streamlit as st
from retriever import get_retriever_chain
# Load environment variables from .env
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set Torch runtime fix from env if not already set
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", os.getenv("TORCH_USE_RTLD_GLOBAL", "1"))

st.set_page_config(page_title="SustainaRAG - CSR/ESG Assistant", page_icon="🌱")

st.markdown(
    """
    <h1 style='text-align: center;'>🌱 SustainaRAG</h1>
    <h3 style='text-align: center; color: grey;'>A RAG-powered assistant for sustainable business strategy design</h3>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_chain():
    return get_retriever_chain()

qa_chain = load_chain()

query = st.text_input("Ask a sustainability-related question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke(query)
        st.markdown("### Answer")
        st.write(result['result'])

        st.markdown("### Top Matching Sources")
        for i, doc in enumerate(result['source_documents']):
            st.markdown(f"**{i+1}. Source:** `{doc.metadata.get('source', 'Unknown')}`")
            st.markdown(f"> {doc.page_content[:300]}...")