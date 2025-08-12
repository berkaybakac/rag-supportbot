import os
from pathlib import Path
from typing import Literal

import faiss
from tqdm import tqdm

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index import StorageContext
from llama_index.vector_stores import FaissVectorStore


def build_index(
    data_dir: str = "data",
    persist_dir: str = "db/faiss_index",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    metric: Literal["l2", "ip"] = "l2",
) -> None:
    """
    Belgeleri okuyup parçalara ayırır, embedding'leri çıkarır ve FAISS'e persist eder.

    metric:
        - "l2"  → IndexFlatL2 (normalizasyon gerekmez, stabil)
        - "ip"  → IndexFlatIP (cosine ≈ IP için embedding'leri normalleştirmeniz gerekir; bu örnekte L2 tercih edilir)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"'{data_dir}' klasörü bulunamadı.")

    print(f"[embedder] Belgeler '{data_dir}/' klasöründen yükleniyor...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        raise RuntimeError(
            "Yüklenecek belge bulunamadı. Lütfen 'data/' içine dosyalar ekleyin."
        )

    print(f"[embedder] Toplam belge: {len(documents)}")
    print(f"[embedder] Embedding modeli yükleniyor: {embedding_model_name}")
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    # Boyutu belirlemek için tek seferlik probe
    probe_vec = embed_model.get_text_embedding("dim_probe")
    dim = len(probe_vec)
    print(f"[embedder] Embedding vektör boyutu: {dim}")

    # FAISS index (varsayılan: L2 – normalizasyon gerektirmez)
    if metric == "ip":
        faiss_index = faiss.IndexFlatIP(dim)
        print("[embedder] FAISS index: IndexFlatIP (inner product)")
    else:
        faiss_index = faiss.IndexFlatL2(dim)
        print("[embedder] FAISS index: IndexFlatL2 (L2 distance)")

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        text_splitter=text_splitter,
    )

    print("[embedder] Indeks oluşturma ve persist başlıyor...")
    os.makedirs(persist_dir, exist_ok=True)
    # Not: from_documents çağrısı, FAISS store'a otomatik olarak yazacaktır.
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    storage_context.persist(persist_dir=persist_dir)
    print(f"[embedder] FAISS index başarıyla kaydedildi → {persist_dir}/")


if __name__ == "__main__":
    # İsterseniz metric="ip" verip (cosine için) kendi normalizasyon stratejinizi ekleyebilirsiniz.
    build_index()
