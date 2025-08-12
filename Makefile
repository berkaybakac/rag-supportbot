.PHONY: emb ret llm uis lst fbk logs watchlogs

emb: ; python -m app.embedder          # embedder çalıştır
ret: ; python -m app.retriever         # retriever çalıştır
llm: ; python -m app.llm_generator     # LLM generator çalıştır

uis:
	PYTHONPATH=. streamlit run app/ui_streamlit.py

# feedback_logger test ve görüntüleme
fbk: ; python -m app.feedback_logger --test --show 3
logs: ; python -m app.feedback_logger --show 20
watchlogs: ; tail -f logs/queries.jsonl || true

lst:                                   # modül listesini göster
	@ls app/*.py 2>/dev/null \
	| sed 's#.*/##;s/\.py$$//' \
	| grep -v '^__init__$$' || true
