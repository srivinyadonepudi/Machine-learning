import streamlit as st
from embedding import NeMoEmbedder
from chunking import load_and_chunk_document
from faiss_index import FaissIndex
from retriever import Retriever
from triton_client import TritonClient
from prompt_builder import build_prompt
import os

INDEX_PATH = "faiss.index"

st.set_page_config(page_title="RAG Chatbot Demo", layout="wide")

@st.cache_resource(ttl=3600)
def load_embedding_model():
    return NeMoEmbedder()

@st.cache_resource(ttl=3600)
def load_faiss_index():
    if os.path.exists(INDEX_PATH + ".index"):
        return FaissIndex.load(INDEX_PATH)
    else:
        return FaissIndex(dim=768)

@st.cache_resource(ttl=3600)
def load_triton_client():
    return TritonClient(url="localhost:8001", model_name="llm_model")

def main():
    st.title("RAG Chatbot with NeMo + FAISS + Triton + Streamlit")

    embedder = load_embedding_model()
    faiss_index = load_faiss_index()
    retriever = Retriever(faiss_index)
    triton_client = load_triton_client()

    # Document upload
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, TXT, DOCX files", accept_multiple_files=True, type=['pdf','txt','docx'])
    if uploaded_files:
        new_chunks = []
        for file in uploaded_files:
            chunks = load_and_chunk_document(file)
            new_chunks.extend(chunks)
        embeddings = embedder.embed_texts([c[1] for c in new_chunks])
        faiss_index.add_embeddings(embeddings, new_chunks)
        faiss_index.save(INDEX_PATH)
        st.sidebar.success(f"Added {len(new_chunks)} chunks to the index!")

    # Chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your documents:", key="input")

    if st.button("Submit") or query:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        query_emb = embedder.embed_texts([query])[0]
        top_chunks = retriever.retrieve(query_emb, top_k=5)
        prompt = build_prompt(query, top_chunks)
        answer = triton_client.infer(prompt)
        st.session_state.chat_history.append({"query": query, "answer": answer, "context": top_chunks})

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {chat['query']}")
        st.markdown(f"**A:** {chat['answer']}")
        with st.expander("Show retrieved context snippets"):
            for idx, (doc_id, chunk_text) in enumerate(chat["context"]):
                st.markdown(f"- **Doc {doc_id} chunk {idx+1}:** {chunk_text[:300]}...")

if __name__ == "__main__":
    main()
