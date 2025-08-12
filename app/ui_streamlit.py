# app/ui_streamlit.py
import os
import streamlit as st

# Tokenizers uyarısını kapat (fork sonrası deadlock uyarıları için)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Streamlit Cloud/Secrets -> env köprüsü (dotenv yoksa)
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

st.set_page_config(page_title="RAG SupportBot", layout="wide")
st.title("RAG SupportBot")
st.markdown("Belgelerinizden beslenen teknik destek asistanı.")

# ───────── Belge yükleme + embedding (lazy import) ─────────
st.subheader("Belge Yükleme")
uploaded_files = st.file_uploader(
    "Belgeleri seçin (TXT, PDF, MD)", accept_multiple_files=True
)

if uploaded_files:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(data_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.info(f"{len(uploaded_files)} belge yüklendi. Embedding başlatılıyor…")
    try:
        from app.embedder import build_index  # lazy import

        build_index(data_dir=data_dir, persist_dir="db/faiss_index")
        st.success("Embedding tamamlandı. Yeni belgeler kullanılabilir.")
        st.session_state["index_ready"] = True
    except Exception as e:
        st.error("Embedding sırasında hata oluştu.")
        with st.expander("Hata detayı"):
            st.code(repr(e))


# ───────── Retriever cache (lazy init) ─────────
@st.cache_resource(show_spinner="Arama motoru hazırlanıyor…")
def load_retriever():
    from app.retriever import DocumentRetriever  # lazy import

    return DocumentRetriever()


# ───────── Soru/cevap ─────────
st.subheader("Soru")
question = st.text_input("Sorunuzu buraya yazın:")

if question:
    if not os.path.isdir("db/faiss_index"):
        st.warning(
            "FAISS indeksi bulunamadı. Lütfen belge yükleyip embedding oluşturun."
        )
        st.stop()

    try:
        retriever = load_retriever()  # cache_resource sayesinde tek kez kurulur

        with st.spinner("Belgeler aranıyor…"):
            results = retriever.retrieve(question) or []

        if not results:
            st.warning("Bu sorunun cevabı elimdeki belgelerde bulunamadı.")
            st.session_state.pop("last_answer", None)
        else:
            with st.spinner("Yanıt üretiliyor…"):
                from app.llm_generator import generate_answer  # lazy import

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

            st.subheader("Kaynak Belgeler")
            for i, node in enumerate(results):
                # farklı düğüm tiplerine dayanıklı okuma
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

# ───────── Geri Bildirim ─────────
if "last_answer" in st.session_state:
    st.subheader("Geri Bildirim")
    with st.form(key=f"feedback_form_{st.session_state.get('feedback_key','0')}"):
        helpful_choice = st.radio(
            "Yanıt faydalı oldu mu?",
            options=("Evet", "Hayır"),
            horizontal=True,
            index=0,
        )
        comment = st.text_input("Yorum (opsiyonel)")
        submitted = st.form_submit_button("Gönder")

    if submitted:
        try:
            from app.feedback_logger import log_feedback

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

            ok, err = log_feedback(
                question=st.session_state.get("last_question", ""),
                answer=st.session_state.get("last_answer", ""),
                helpful=(helpful_choice == "Evet"),
                comment=(comment.strip() or None),
                model=st.session_state.get("last_model"),
                docs=docs_meta,
                extra={"ui": "streamlit"},
            )
            if ok:
                st.success("Geri bildiriminiz kaydedildi. Teşekkürler.")
                # Çift kayıt olmaması için form key değiştir
                st.session_state["feedback_key"] += "_done"
            else:
                st.error(f"Geri bildirim kaydı başarısız: {err}")
        except Exception as e:
            st.error("Geri bildirim kaydedilemedi.")
            with st.expander("Hata detayı"):
                st.code(repr(e))

# ───────── Geçici debug panel (sorun giderme için) ─────────
with st.expander("Debug (geçici)"):
    st.write("CWD:", os.getcwd())
    try:
        st.write("Repo kökü:", os.listdir("."))
        st.write("app/:", os.listdir("app"))
        st.write("db/faiss_index var mı?:", os.path.isdir("db/faiss_index"))
    except Exception as e:
        st.write("listdir error:", repr(e))
    st.write(
        {
            "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL"),
            "SYSTEM_PROMPT_PATH": os.getenv("SYSTEM_PROMPT_PATH"),
        }
    )
