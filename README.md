# RAG Chatbot with NVIDIA NeMo + FAISS + Triton + Streamlit

This is a Retrieval-Augmented Generation (RAG) chatbot that:

- Embeds documents using NVIDIA NeMo text embedding models
- Stores embeddings in a FAISS vector index
- Retrieves relevant chunks for queries
- Sends context+query prompt to an LLM served by NVIDIA Triton Inference Server
- Provides a Streamlit UI for interactive Q&A

## Setup

```bash
pip install -r requirements.txt
