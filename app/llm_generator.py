# app/llm_generator.py

import os
import sys
import requests
from typing import List, Optional
from llama_index.schema import NodeWithScore
from dotenv import load_dotenv

# Proje kök klasörünü modül olarak tanıtma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ortam değişkenlerini yükle (.env)
load_dotenv()

# OpenRouter ayarları
API_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")

# Sistem Prompt'u
SYSTEM_PROMPT = (
    "Sen bir teknik destek yapay zeka asistanısın. Görevin, YALNIZCA sana sağlanan belgelerden yola çıkarak "
    "kullanıcı sorusunu cevaplamaktır.\n"
    "ASLA belge dışında bilgi verme, varsayımda bulunma veya yorum yapma.\n"
    "Cevabının sonunda, kullandığın belge referansını [Kaynak: Belge Adı] şeklinde belirt.\n"
    "Eğer sorunun cevabı sağlanan belgelerde yoksa, kibarca 'Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır.' diye yanıt ver."
)


def generate_answer(
    question: str,
    contexts: List[NodeWithScore],
    model_name: Optional[str] = None,
) -> str:
    """
    model_name parametresi verilirse onu kullanır; verilmezse .env'deki OPENROUTER_MODEL'i,
    o da yoksa DEFAULT_MODEL sabitini kullanır.
    """
    if not API_KEY:
        raise ValueError(
            "API anahtarı bulunamadı. .env içindeki OPENROUTER_API_KEY değerini kontrol et."
        )

    model = model_name or DEFAULT_MODEL

    # Belge parçalarını birleştir
    context_text = "\n\n".join(
        [f"[Belge {i+1}]:\n{node.text.strip()}" for i, node in enumerate(contexts)]
    )

    # Kullanıcıya gidecek prompt
    prompt = f"Soru: {question}\n\nBelgeler:\n{context_text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 768,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter bazı durumlarda referer ve/veya başlık ister; referansı koruyoruz
        # çünkü bu repo GitHub'da barındırılıyor.
        "HTTP-Referer": "https://github.com/berkaybakac/rag-supportbot",
        "X-Title": "rag-supportbot",
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise Exception(f"API hatası: {resp.status_code} - {resp.text}")

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except Exception:
        raise Exception(f"Beklenmeyen API yanıtı: {data}")

    return answer.strip()


if __name__ == "__main__":
    from app.retriever import DocumentRetriever

    retriever = DocumentRetriever()
    question = input("Soru: ")
    results = retriever.retrieve(question)

    print("\n--- En Benzer Belgeler ---")
    for i, node in enumerate(results):
        print(f"[{i+1}] Skor: {node.score:.4f}")
        print(node.text[:200], "\n")

    print("\n--- LLM Cevabı ---")
    # Burada model adı vermiyoruz; .env'den okunacak
    answer = generate_answer(question, results)

    with open("last_answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    print("Yanıt dosyaya yazıldı → last_answer.txt")
