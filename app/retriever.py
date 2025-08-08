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


class DocumentRetriever:
    def __init__(
        self,
        persist_dir: str = "db/faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        top_k: int = 3,
    ):
        self.top_k = top_k
        self.persist_dir = persist_dir

        # Embedding modeli
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

        # Chunk yapısı (gerekirse yeniden kullanılır)
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)

        self.service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            text_splitter=text_splitter,
        )

        # FAISS index'i yükle
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(
            storage_context=storage_context,
            service_context=self.service_context,
        )

        # Sorgulayıcı (query engine)
        self.query_engine = self.index.as_query_engine(similarity_top_k=top_k)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Verilen sorguya en uygun belge parçalarını döndürür"""
        results = self.query_engine.retrieve(query)
        return results


if __name__ == "__main__":
    retriever = DocumentRetriever()
    query = input("Soru girin: ")
    results = retriever.retrieve(query)

    print("\nEn Benzer Belgeler:")
    for i, node in enumerate(results):
        print(f"\n--- Parça {i+1} ---")
        print(node.text[:500])
        print(f"\nSkor: {node.score:.4f}")
