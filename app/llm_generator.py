# app/llm_generator_openrouter.py

import os
import sys
import requests
from typing import List
from llama_index.schema import NodeWithScore
from dotenv import load_dotenv

# Proje kök klasörünü modül olarak tanıtma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ortam değişkenlerini yükle (.env)
load_dotenv()

# OpenRouter ayarları
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Sistem Prompt'unu biraz daha kesin hale getirelim
SYSTEM_PROMPT = (
    "Sen bir teknik destek yapay zekasısın. Görevin, YALNIZCA sana sağlanan belgelerden yola çıkarak kullanıcı sorusunu cevaplamaktır.\n"
    "ASLA belge dışında bilgi verme, varsayımda bulunma veya yorum yapma.\n"
    "Cevabının sonunda, kullandığın belge referansını [Kaynak: Belge Adı] şeklinde belirt.\n"
    "Eğer sorunun cevabı sağlanan belgelerde yoksa, kibarca 'Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır.' diye yanıt ver."
)


def generate_answer(
    question: str,
    contexts: List[NodeWithScore],
    model_name: str = "meta-llama/llama-3-8b-instruct",
) -> str:
    if not API_KEY:
        raise ValueError("API anahtarı bulunamadı. .env dosyasını kontrol et.")

    # Chunk'ları birleştir
    context_text = "\n\n".join(
        [f"[Belge {i+1}]:\n{node.text.strip()}" for i, node in enumerate(contexts)]
    )

    # Prompt
    prompt = f"Soru: {question}\n\nBelgeler:\n{context_text}"

    # OpenRouter body
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,  # Daha az 'yaratıcı' ve daha gerçekçi bir cevap için sıcaklığı düşürelim
        "max_tokens": 768,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/berkaybakac/rag-supportbot",
    }

    # İstek
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API hatası: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


# Test
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
    answer = generate_answer(question, results)

    with open("last_answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    print("Yanıt dosyaya yazıldı → last_answer.txt")
