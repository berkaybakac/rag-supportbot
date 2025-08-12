# app/feedback_logger.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional
from collections import deque
import argparse

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "queries.jsonl")


def _ensure_logfile() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")


def _extract_node_info(node: Any) -> dict:
    # LlamaIndex sürümlerine karşı defansif okuma
    meta = getattr(node, "metadata", None)
    if meta is None and hasattr(node, "node"):
        meta = getattr(node.node, "metadata", None)

    node_id = getattr(node, "node_id", None)
    if node_id is None and hasattr(node, "node"):
        node_id = getattr(node.node, "node_id", None)

    score = getattr(node, "score", None)
    if score is None:
        score = getattr(node, "similarity", None)

    text = getattr(node, "text", None)
    if text is None and hasattr(node, "node"):
        text = getattr(node.node, "text", None)

    src = None
    if isinstance(meta, dict):
        src = meta.get("file_path") or meta.get("source") or meta.get("doc_id")

    return {
        "node_id": node_id,
        "score": score,
        "source": src,
        "preview": (text or "")[:300],
    }


def log_feedback(
    *,
    question: str,
    answer: str,
    nodes: Iterable[Any],
    helpful: bool,
    comment: str = "",
    model: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Geri bildirimi JSONL satırı olarak kaydeder."""
    _ensure_logfile()

    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "helpful": bool(helpful),
        "comment": comment or "",
        "model": model,
        "docs": [_extract_node_info(n) for n in (nodes or [])],
    }
    if extra:
        payload["extra"] = extra

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_last(n: int = 20) -> list[dict]:
    """Log dosyasından son n kaydı döndürür."""
    _ensure_logfile()
    rows: deque[str] = deque(maxlen=n)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line)
    return [json.loads(x) for x in rows]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Feedback logger CLI")
    parser.add_argument("--show", type=int, default=0, help="Son N kaydı göster.")
    parser.add_argument(
        "--test", action="store_true", help="Test amaçlı örnek kayıt ekle."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Soru (isteğe bağlı, CLI ile log yazmak için).",
    )
    parser.add_argument(
        "--answer",
        "-a",
        type=str,
        help="Cevap (isteğe bağlı, CLI ile log yazmak için).",
    )
    parser.add_argument(
        "--helpful", choices=["yes", "no"], default="yes", help="Faydalı mı?"
    )
    parser.add_argument("--comment", type=str, default="", help="Yorum (opsiyonel).")
    args = parser.parse_args()

    wrote = False

    if args.test:
        log_feedback(
            question="Test sorusu",
            answer="Test cevabı",
            nodes=[],
            helpful=True,
            comment="CLI test kaydı",
            model=os.getenv("OPENROUTER_MODEL"),
            extra={"cli": True},
        )
        print("Test kaydı eklendi ->", LOG_FILE)
        wrote = True

    if args.question and args.answer:
        log_feedback(
            question=args.question,
            answer=args.answer,
            nodes=[],
            helpful=(args.helpful == "yes"),
            comment=args.comment,
            model=os.getenv("OPENROUTER_MODEL"),
            extra={"cli": True},
        )
        print("Kayıt eklendi ->", LOG_FILE)
        wrote = True

    if args.show or not wrote:
        # Hiç yazma yapılmadıysa varsayılan olarak son 10 kaydı gösterelim
        n = args.show if args.show else 10
        items = read_last(n)
        if not items:
            print("Log boş:", LOG_FILE)
            return
        print(f"Son {len(items)} kayıt (dosya: {LOG_FILE}):")
        for i, row in enumerate(items, 1):
            print(f"\n[{i}] {row.get('ts')}")
            print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
