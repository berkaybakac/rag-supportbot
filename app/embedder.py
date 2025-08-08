import os
from pathlib import Path
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.text_splitter import SentenceSplitter
from tqdm import tqdm


def build_index(
    data_dir: str = "data",
    persist_dir: str = "db/faiss_index",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
):
    print(f"Belgeler '{data_dir}/' klasöründen yükleniyor...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Toplam belge: {len(documents)}")

    print(f"Embedding modeli yükleniyor: {embedding_model_name}")
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        text_splitter=text_splitter,
    )

    print("Belge embedding işlemi başlıyor...")
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        show_progress=True,
    )

    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)

    print(f"FAISS index başarıyla kaydedildi: {persist_dir}/")


if __name__ == "__main__":
    build_index()
