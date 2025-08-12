# app/ui_streamlit.py
import os
import glob
from typing import List
import streamlit as st

# Uyarıyı kapat
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

st.set_page_config(page_title="RAG SupportBot", layout="wide")
st.title("RAG SupportBot")
st.markdown("Belgelerinizden beslenen teknik destek asistanı.")


# --- Yardımcı: indeks imzası (değişince cache bozulsun) ---
def index_signature(path: str = "db/faiss_index") -> str:
    if not os.path.isdir(path):
        return "no-index"
    mtimes: List[str] = []
    for p in sorted(glob.glob(os.path.join(path, "**"), recursive=True)):
        if os.path.isfile(p):
            try:
                mtimes.append(f"{p}:{os.path.getmtime(p)}")
            except Exception:
                pass
    return "|".join(mtimes) or "empty-index"


# ───────── Belge yükleme + embedding (lazy import) ─────────
st.subheader("Belge Yükleme")
uploaded_files = st.file_uploader(
    "Belgeleri seçin (TXT, PDF, MD)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
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
        # İndeks değişti; retriever cache'ini kırmak için sayfayı yenile
        st.session_state["index_ready"] = True
        st.rerun()
    except Exception as e:
        st.error("Embedding sırasında hata oluştu.")
        with st.expander("Hata detayı"):
            st.code(repr(e))


# ───────── Retriever cache (indeks imzasına bağlı) ─────────
@st.cache_resource(show_spinner="Arama motoru hazırlanıyor…")
def load_retriever(sig: str):
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
        retriever = load_retriever(index_signature())  # indeks değişince cache bozulur
        with st.spinner("Belgeler aranıyor…"):
            results = retriever.retrieve(question) or []

        if not results:
            st.warning("Bu sorunun cevabı elimdeki belgelerde bulunamadı.")
        else:
            with st.spinner("Yanıt üretiliyor…"):
                from app.llm_generator import generate_answer  # lazy import

                answer = generate_answer(question, results)

            st.subheader("Yanıt")
            st.write(answer)

            st.subheader("Kaynak Belgeler")
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

# ───────── Geçici debug panel ─────────
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
