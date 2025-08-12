from pathlib import Path
from typing import List

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import NodeWithScore

# 0.9.x için:
from llama_index.indices.query.schema import QueryBundle
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import FaissVectorStore


class DocumentRetriever:
    """
    Persist edilmiş FAISS index'i yükler, sorgu için retriever + reranker sunar.
    """

    def __init__(
        self,
        persist_dir: str = "db/faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name: str = "BAAI/bge-reranker-base",
        chunk_size: int = 500,
        top_k_retrieval: int = 10,  # ilk getirilen parça sayısı
        top_k_rerank: int = 2,  # rerank sonrası saklanacak parça sayısı
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        # Embedding + splitter + service context
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            text_splitter=self.text_splitter,
        )

        # FAISS VectorStore'ı persistten yükle
        if not Path(persist_dir).exists():
            raise FileNotFoundError(
                f"Persist klasörü '{persist_dir}' bulunamadı. Önce embedder.py çalıştırın."
            )
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=vector_store,
        )

        # Index'i yükle
        self.index: VectorStoreIndex = load_index_from_storage(
            storage_context=storage_context,
            service_context=self.service_context,
        )

        # Retriever + Reranker + QueryEngine
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k_retrieval)
        self.reranker = SentenceTransformerRerank(
            model=rerank_model_name,
            top_n=self.top_k_rerank,
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever,
            node_postprocessors=[self.reranker],
        )

    def retrieve(self, query: str) -> List[NodeWithScore]:
        query_bundle = QueryBundle(query_str=query)
        results: List[NodeWithScore] = self.query_engine.retrieve(query_bundle)
        return results


if __name__ == "__main__":
    retriever = DocumentRetriever()
    query = input("Soru girin: ").strip()
    results = retriever.retrieve(query)

    if not results:
        print("Sonuç bulunamadı.")
    else:
        print("\nEn Benzer ve Yeniden Sıralanmış Belgeler:")
        for i, node in enumerate(results):
            print(f"\n--- Parça {i+1} ---")
            print(node.text)
            print(f"\nSkor: {node.score:.4f}")
