import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.schema import NodeWithScore

# İsteğe bağlı: HF tokenizers uyarısını kapatmak istersen .env'de zaten ayarlı olabilir.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

API_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")

# Varsayılan sistem promptu (dosya okunamazsa buna düşer)
DEFAULT_SYSTEM_PROMPT = (
    "Sen bir teknik destek yapay zeka asistanısın. Görevin, YALNIZCA sana sağlanan belgelerden yola çıkarak "
    "kullanıcı sorusunu cevaplamaktır.\n"
    "ASLA belge dışında bilgi verme, varsayımda bulunma veya yorum yapma.\n"
    "Cevabının sonunda, kullandığın belge referansını [Kaynak: Belge N] şeklinde belirt.\n"
    "Eğer sorunun cevabı sağlanan belgelerde yoksa, kibarca 'Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır.' diye yanıt ver."
)

PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    os.path.join(os.path.dirname(__file__), "prompts", "system_v1.txt"),
)


def load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                raise ValueError("Boş prompt dosyası")
            return txt
    except Exception as e:
        print(
            f"Sistem promptu dosyadan okunamadı ({e}). Varsayılan prompt kullanılacak."
        )
        return DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = load_system_prompt()


def _node_label(i: int, node: NodeWithScore) -> str:
    """
    Kaynak bilgisini etiket olarak ekler (varsa).
    """
    meta = getattr(node, "metadata", None)
    if not meta and hasattr(node, "node"):
        meta = getattr(node.node, "metadata", None)
    src = None
    if isinstance(meta, dict):
        src = meta.get("file_path") or meta.get("source") or meta.get("doc_id")
    label = f"[Belge {i+1}"
    if src:
        import os as _os

        label += f" | {_os.path.basename(str(src))}"
    label += "]"
    return label


def generate_answer(
    question: str,
    contexts: List[NodeWithScore],
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 768,
) -> str:
    if not API_KEY:
        raise ValueError(
            "API anahtarı bulunamadı. .env içindeki OPENROUTER_API_KEY değerini kontrol et."
        )

    # Boş sonuç güvenliği: belge yoksa LLM'i çağırma
    if not contexts:
        return "Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır."

    model = model_name or DEFAULT_MODEL

    # Belgeleri tek metinde birleştir (kaynak etiketleriyle)
    chunks: List[str] = []
    for i, node in enumerate(contexts):
        node_text = getattr(node, "text", None) or getattr(
            getattr(node, "node", None), "text", ""
        )
        node_text = (node_text or "").strip()
        if not node_text:
            continue
        label = _node_label(i, node)
        chunks.append(f"{label}:\n{node_text}")
    context_text = "\n\n".join(chunks).strip()

    if not context_text:
        return "Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır."

    user_prompt = f"Soru: {question}\n\nBelgeler:\n{context_text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/berkaybakac/rag-supportbot",
        "X-Title": "rag-supportbot",
    }

    print("OpenRouter model =", payload["model"])

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        try:
            j = resp.json()
        except Exception:
            j = {"raw": resp.text}
        raise RuntimeError(f"API hatası: {resp.status_code} - {json.dumps(j)[:800]}")

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Beklenmeyen API yanıtı: {json.dumps(data)[:800]}") from e

    if not answer:
        answer = "Bu sorunun cevabı elimdeki belgelerde bulunmamaktadır."
    return answer


if __name__ == "__main__":
    from app.retriever import DocumentRetriever  # DÜZELTİLDİ

    retriever = DocumentRetriever()
    question = input("Soru: ").strip()
    results = retriever.retrieve(question)

    print("\n--- En Benzer Belgeler (Rerank sonrası) ---")
    if not results:
        print("(boş)")
    else:
        for i, node in enumerate(results):
            node_preview = (getattr(node, "text", "") or "")[:200].replace("\n", " ")
            print(f"[{i+1}] Skor: {node.score:.4f} | {node_preview}...")

    print("\n--- LLM Cevabı ---")
    answer = generate_answer(question, results)
    print(answer)

    with open("last_answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)
    print("Yanıt dosyaya yazıldı → last_answer.txt")
