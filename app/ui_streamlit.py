import os
import json
import uuid
import streamlit as st
from app.feedback_logger import LOG_PATH as FB_LOG_PATH, log_feedback

# Gürültüyü kapat
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Streamlit Cloud secrets -> env köprüsü
for k in (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "SYSTEM_PROMPT_PATH",
    "TOKENIZERS_PARALLELISM",
):
    try:
        if hasattr(st, "secrets") and k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


# ----------------------------- Yardımcılar -----------------------------
@st.cache_data(ttl=30)
def _read_feedback(path: str = FB_LOG_PATH, limit: int = 50):
    """JSONL logundan son N kaydı okunur ve tablo görünümü hazırlanır."""
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    rows = rows[-limit:]
    view = []
    for r in reversed(rows):  # en yeni üstte
        view.append(
            {
                "Zaman": r.get("ts", ""),
                "Takip No": r.get("id", "")
                or r.get("extra", {}).get("feedback_id", ""),
                "Faydalı?": "Evet" if r.get("helpful") else "Hayır",
                "Yorum": r.get("comment", "") or "",
                "Model": r.get("model", "") or "",
                "Doc#": len(r.get("docs", []) or []),
                "Soru (ilk 120)": (r.get("question", "") or "")[:120],
            }
        )
    return view


def _clear_feedback_cache():
    try:
        _read_feedback.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


# ------------------------------- UI -------------------------------
st.set_page_config(page_title="RAG SupportBot", layout="wide")
st.title("RAG SupportBot")
st.caption("Belgelerinizden beslenen teknik destek asistanı.")

# Belge yükleme + embedding
with st.expander("Belge Yükleme (TXT, PDF, MD)"):
    uploaded_files = st.file_uploader("Dosyaları yükleyin", accept_multiple_files=True)
    if uploaded_files:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(data_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.info(f"{len(uploaded_files)} belge yüklendi. Embedding başlatılıyor…")
        try:
            from app.embedder import build_index

            build_index(data_dir=data_dir, persist_dir="db/faiss_index")
            st.success("Embedding tamamlandı. Yeni belgeler kullanılabilir.")
            st.session_state["index_ready"] = True
        except Exception as e:
            st.error("Embedding sırasında hata oluştu.")
            with st.expander("Hata detayı"):
                st.code(repr(e))

# Soru / Cevap
st.subheader("Soru")
question = st.text_input(
    "Sorunuzu yazın ve Enter'a basın:",
    placeholder="Örn: Makine sıkmada gürültü yapıyor, nereleri kontrol etmeliyim?",
)

if question:
    if not os.path.isdir("db/faiss_index"):
        st.warning(
            "FAISS indeksi bulunamadı. Lütfen belge yükleyip embedding oluşturun."
        )
        st.stop()

    try:

        @st.cache_resource(show_spinner=False)
        def _load_retriever():
            from app.retriever import DocumentRetriever

            return DocumentRetriever()

        retriever = _load_retriever()

        with st.spinner("Belgeler aranıyor…"):
            results = retriever.retrieve(question) or []

        if not results:
            st.warning("Bu sorunun cevabı elimdeki belgelerde bulunamadı.")
            st.session_state.pop("last_answer", None)
        else:
            with st.spinner("Yanıt üretiliyor…"):
                from app.llm_generator import generate_answer

                answer = generate_answer(question, results)

            st.subheader("Yanıt")
            st.write(answer)

            # Feedback için state
            st.session_state["last_question"] = question
            st.session_state["last_answer"] = answer
            st.session_state["last_nodes"] = results
            st.session_state["last_model"] = os.getenv("OPENROUTER_MODEL")
            st.session_state["feedback_key"] = str(
                abs(hash(question + "\n" + answer)) % (10**12)
            )

            with st.expander("Kaynak Belgeler"):
                for i, node in enumerate(results):
                    text = getattr(node, "text", None)
                    if text is None and hasattr(node, "node"):
                        text = getattr(node.node, "text", "")
                    text = (text or "")[:400]
                    score = getattr(node, "score", getattr(node, "similarity", 0.0))
                    st.markdown(f"**Belge {i+1}** (Skor: {score:.4f})\n\n{text}...")
    except Exception as e:
        st.error("Sorgu çalıştırılırken hata oluştu.")
        with st.expander("Hata detayı"):
            st.code(repr(e))

# Geri Bildirim
if "last_answer" in st.session_state:
    st.subheader("Geri Bildirim")
    with st.form(key=f"feedback_form_{st.session_state.get('feedback_key','0')}"):
        col1, col2 = st.columns([1, 2])
        with col1:
            helpful_choice = st.radio(
                "Yanıt faydalı oldu mu?",
                options=("Evet", "Hayır"),
                horizontal=True,
                index=0,
            )
        with col2:
            comment = st.text_input(
                "Yorum (opsiyonel)", placeholder="Kısa yorum bırakabilirsiniz…"
            )
        submitted = st.form_submit_button("Gönder", use_container_width=True)

    if submitted:
        try:
            # Kaynak meta listesi
            docs_meta = []
            for node in st.session_state.get("last_nodes", []) or []:
                meta = {}
                if hasattr(node, "node") and getattr(node.node, "metadata", None):
                    meta = dict(node.node.metadata or {})
                elif getattr(node, "metadata", None):
                    meta = dict(node.metadata or {})
                docs_meta.append(
                    {
                        "score": getattr(
                            node, "score", getattr(node, "similarity", None)
                        ),
                        "source": meta.get("file_path")
                        or meta.get("source")
                        or meta.get("doc_id"),
                    }
                )

            feedback_id = uuid.uuid4().hex[:10]
            ok, err, fid = log_feedback(
                question=st.session_state.get("last_question", ""),
                answer=st.session_state.get("last_answer", ""),
                helpful=(helpful_choice == "Evet"),
                comment=(comment.strip() or None),
                model=st.session_state.get("last_model"),
                docs=docs_meta,
                extra={"ui": "streamlit"},
                fb_id=feedback_id,
            )

            if ok:
                st.success(f"Geri bildiriminiz kaydedildi. Takip No: {fid}")
                st.session_state["feedback_key"] += "_done"
                _clear_feedback_cache()
                st.rerun()  # tabloyu anında yenile
            else:
                st.error(f"Geri bildirim kaydı başarısız: {err}")
        except Exception as e:
            st.error("Geri bildirim kaydedilemedi.")
            with st.expander("Hata detayı"):
                st.code(repr(e))

# Geri Bildirim Geçmişi
st.subheader("Geri Bildirim Geçmişi")
colA, _ = st.columns([1, 6])
with colA:
    if st.button("Yenile"):
        _clear_feedback_cache()
        st.rerun()

_fb_rows = _read_feedback()
if not _fb_rows:
    st.caption("Henüz geri bildirim kaydı yok.")
else:
    st.dataframe(_fb_rows, use_container_width=True, hide_index=True)
    try:
        with open(FB_LOG_PATH, "rb") as f:
            st.download_button(
                "Log dosyasını indir (JSONL)",
                f,
                file_name="queries.jsonl",
                mime="application/json",
            )
    except Exception:
        pass

# Geliştirici bölümü (isteğe bağlı kısa)
with st.expander("Debug"):
    st.write("CWD:", os.getcwd())
    st.write("Log yolu:", FB_LOG_PATH)
    st.write("db/faiss_index var mı?:", os.path.isdir("db/faiss_index"))
