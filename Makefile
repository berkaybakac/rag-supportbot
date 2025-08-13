.PHONY: emb ret llm uis fbk logs watchlogs lst

# Embedding (FAISS index oluştur/güncelle)
emb: ; python -m app.embedder

# Retriever CLI
ret: ; python -m app.retriever

# LLM generator CLI
llm: ; python -m app.llm_generator

# Streamlit UI
uis:
	PYTHONPATH=. streamlit run app/ui_streamlit.py

# Feedback test + son 3 kayıt
fbk: ; python -m app.feedback_logger --test --show 3

# Son 20 log kaydı
logs: ; python -m app.feedback_logger --show 20

# Log dosyasını canlı izle (yoksa oluştur)
watchlogs:
	@mkdir -p logs
	@[ -f logs/queries.jsonl ] || : > logs/queries.jsonl
	tail -f logs/queries.jsonl || true

# Modül listesini göster
lst:
	@ls app/*.py 2>/dev/null \
	| sed 's#.*/##;s/\.py$$//' \
	| grep -v '^__init__$$' || true
