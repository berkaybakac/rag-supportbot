from __future__ import annotations
import os, json, argparse
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Iterable

# Repo köküne göre mutlak log yolu (UI/CLI nereden çalışırsa çalışsın aynı dosyaya yazar)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "logs", "queries.jsonl")
LOG_PATH = os.getenv("RAG_LOG_PATH", DEFAULT_LOG_PATH)


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def log_feedback(
    *,
    question: str,
    answer: str,
    helpful: bool,
    comment: str | None = None,
    model: str | None = None,
    docs: Iterable[dict] | None = None,
    extra: dict | None = None,
    email: str | None = None,
    fb_id: str | None = None,
    path: str = LOG_PATH,
) -> tuple[bool, str | None, str]:
    """JSONL satırı olarak kaydeder. (ok, err, fb_id) döndürür."""
    _ensure_dir(path)
    fb_id = (
        fb_id or f"FB-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    )
    row = {
        "id": fb_id,
        "ts": _now_iso(),
        "question": question,
        "answer": answer,
        "helpful": bool(helpful),
        "comment": comment,
        "model": model,
        "docs": list(docs or []),
        "extra": extra or {},
        "email": email,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True, None, fb_id
    except Exception as e:
        return False, repr(e), fb_id


def tail(path: str = LOG_PATH, n: int = 20) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[-n:]


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--show", type=int, default=0, help="Son N kaydı göster")
    p.add_argument("--test", action="store_true", help="Test kaydı ekle")
    args = p.parse_args()

    if args.test:
        ok, err, fid = log_feedback(
            question="Test sorusu",
            answer="Test cevabı",
            helpful=True,
            comment="CLI test kaydı",
            docs=[],
            extra={"cli": True},
        )
        print(f"Test kaydı eklendi -> {LOG_PATH}")
    if args.show:
        rows = tail(n=args.show)
        print(f"Son {len(rows)} kayıt (dosya: {LOG_PATH}):\n")
        for i, r in enumerate(rows, 1):
            print(f"[{i}] {r.get('ts','')}  id={r.get('id','')}")
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print()


if __name__ == "__main__":
    _cli()
