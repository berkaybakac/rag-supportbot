# RAG-SupportBot: Mimari Dokümantasyon

Bu belge, cihaz üstünde (offline) veya online çalışan, belge tabanlı teknik destek asistanı olan **RAG-SupportBot** projesinin mimari kararlarını, kullanılan teknolojileri ve bileşenlerin işleyişini detaylı şekilde açıklamak amacıyla hazırlanmıştır.

## Kullanılan Ana Modeller

### Embedding Modeli: `sentence-transformers/all-MiniLM-L6-v2`

Bu model, küçük boyutlu, hızlı ve semantic search kalitesi açısından dengeli sonuçlar verir.  
MVP aşamasında **yüksek hız** ve **düşük kaynak tüketimi** sağladığı için tercih edilmiştir.  
Daha sonra `nomic-embed-text-v1` veya `instructor-xl` gibi daha güçlü embedding modellerine geçilmesi planlanmaktadır.

### LLM Modeli: `meta-llama/llama-3-8b-instruct`

- **Çalışma şekli:** OpenRouter API üzerinden online olarak çalıştırılır.  
- **Neden seçildi:** RAM sınırlamaları nedeniyle LLM yerelde çalıştırılmamış, API kullanımı ile donanım kısıtı aşılmıştır.  
- **Avantaj:** Farklı LLM modellerine hızlı geçiş imkanı.  
- **Geliştirme önerisi:** Model adı `.env` üzerinden yönetilebilir hale getirilmeli.

## Vector Store: FAISS + LlamaIndex

FAISS, yüksek performanslı bir vektör benzerlik arama kütüphanesidir.  
LlamaIndex ile entegre edilerek belge parçalarının vektörleştirilmiş halleri saklanır ve sorgulara göre en benzer içerikler bulunur.  
Veriler `db/faiss_index` klasöründe JSON tabanlı formatta saklanır.

## MVP Modülleri

### `embedder.py`

- `data/` klasöründeki belgeleri okur.
- Belgeleri **500 token**’lık anlamlı parçalara ayırır.
- Her parçanın embedding vektörünü üretir.
- FAISS vektör veritabanına kaydeder.
- Data değiştiğinde tekrar çalıştırılarak FAISS güncellenir.

### `retriever.py`

- Kullanıcıdan gelen sorunun embedding vektörünü üretir.
- FAISS veritabanından **Top-K** en benzer içerikleri getirir.
- **Reranker** ile bu parçalar, sorguya göre yeniden sıralanır.
- Yalnızca en alakalı içerikler LLM’e iletilir.
- Yanlış cevap verme riski azalır, doğruluk artar.

### `llm_generator.py`

- Retriever’dan gelen belgeler ve kullanıcı sorusunu alır.
- **OpenRouter API** aracılığıyla LLM’e gönderir.
- Prompt yalnızca verilen belgelerden cevap üretmeyi garanti edecek şekilde tasarlanır.
- Üretilen cevap `last_answer.txt` dosyasına kaydedilir.
- **İyileştirme önerisi:** Prompt versiyonlama eklenerek değişiklikler takip edilebilir.

## Gelecek Genişletmeler

- `query_rewriter.py`: Eksik veya kısa soruları iyileştirerek daha iyi arama sonuçları üretir.
- `document_grader.py`: Getirilen belgelerin kalite kontrolünü yapar.
- `fallback_router.py`: Cevap bulunamazsa alternatif işlem akışlarını yönetir.
- Prompt versiyonlama: Prompt değişikliklerinin ayrı dosyada tutulması.

---

## Son Yapılan Ana İyileştirmeler

- **LLM Model Yönetimi (.env üzerinden):**  
  Artık kullanılan LLM modeli kod içine sabit yazılmak yerine `.env` dosyasındaki `OPENROUTER_MODEL` değişkeni üzerinden yönetilmektedir. Böylece model değişikliği için kodu değiştirmeye gerek kalmaz.

- **Prompt Versiyonlama:**  
  Sistem prompt’u, `app/prompts/system_v1.txt` dosyasına taşınarak koddan ayrılmıştır. Prompt değişiklikleri dosya üzerinden yapılabilir, böylece geçmiş versiyonlar takip edilebilir.


## Son Yapılan Ana İyileştirmeler (Güncel)

- **Gerçek FAISS entegrasyonu:** `FaissVectorStore` ile kalıcı vektör deposu kuruldu; embedder FAISS’e indeks yazıyor, retriever FAISS’ten yüklüyor (IndexFlatL2).  
- **Reranker eklendi:** `BAAI/bge-reranker-base` ile ilk getirilen parçalar yeniden sıralanıyor; `top_k_retrieval` ve `top_k_rerank` ayarlanabilir.  
- **LLM çağrısı sertleştirildi:** Boş bağlamda LLM çağrısı yapılmıyor; OpenRouter hata yakalama ve mesaj şeffaflaştırma eklendi.  
- **Prompt dışsallaştırma:** Sistem prompt’u dosyaya alındı; `.env` ile `OPENROUTER_MODEL` değiştirilebilir.  
- **UI’ı modülerleştirme:** Streamlit arayüzü yalnızca **retriever** ve **LLM** katmanlarını çağırıyor; embedder sadece veri güncellenince çalıştırılıyor.  
- **Geri bildirim altyapısı:** `feedback_logger.py` ile JSONL log; UI’dan “Evet/Hayır + yorum” toplanıp anında tabloya yansıtılıyor.  
- **Log yolu kararlılığı:** UI/CLI fark etmeksizin tek dosyaya yazmak için log yolu repo köküne göre **mutlak** hale getirildi.  
- **Makefile hedefleri:** `emb`, `ret`, `llm`, `uis`, `fbk`, `logs`, `watchlogs` ile uçtan uca test akışı.  
- **Repo hijyeni:** `.gitignore` ile FAISS verisi, loglar ve model cache dışlandı; dizinler `.gitkeep` ile korunuyor.

---

## Adım 4: Arayüz ve Geri Bildirim

### 4.1 Basit Web Arayüzü (`app/ui_streamlit.py`)
- Streamlit ile başlık + soru girişi (`st.text_input`) ve sonuç alanı.
- Dosya yükleme expander’ı; yüklenen dosyalar `data/` altına kaydedilip **embedder** tetiklenebilir.

### 4.2 Retriever + LLM Entegrasyonu
- `DocumentRetriever.retrieve()` ile FAISS’ten benzer parçalar alınır.
- `generate_answer()` ile sadece seçili parçalar LLM’e verilir.
- Yanıt ve kaynak önizlemeleri arayüzde gösterilir.

### 4.3 Geri Bildirim Toplama ve Loglama
- Yanıt altında “Yanıt faydalı oldu mu? (Evet/Hayır)” ve “Yorum” alanı.
- `log_feedback()` ile şu şema ile `logs/queries.jsonl` dosyasına ek satır yazılır:
  ```json
  {
    "id": "FB-20250812-AB12CD",
    "ts": "2025-08-12T10:15:30.123456Z",
    "question": "...",
    "answer": "...",
    "helpful": true,
    "comment": "Kısa yorum",
    "model": "meta-llama/llama-3-8b-instruct",
    "docs": [{"score": 0.27, "source": "ariza_kodlari.txt"}],
    "extra": {"ui": "streamlit"}
  }
