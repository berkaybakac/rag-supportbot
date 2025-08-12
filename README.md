# rag-supportbot

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-success)](https://rag-supportbot.streamlit.app)

**rag-supportbot** is a local, document-based AI assistant designed to help technical support teams answer user questions based on existing documents.

The system is built using the Retrieval-Augmented Generation (RAG) architecture, which enables reliable and source-grounded responses by retrieving relevant document chunks and generating answers using a local large language model.

## Key Features

- Offline operation with no external API dependencies
- Source-based, grounded answer generation
- Local vector search using FAISS and LlamaIndex
- Modular architecture for future extensions

## Technologies (initial setup)

- Python 3.11
- FAISS
- LlamaIndex
- LangChain (optional)
- LLaMA 3 Instruct (via `llama.cpp`)
- Streamlit (for UI)

## License

MIT License

> This project is in active development. The architecture and components may change as the 
system evolves.
