.PHONY: emb ret llm uis lst

emb: ; python -m app.embedder          # embedder çalıştır
ret: ; python -m app.retriever         # retriever çalıştır
llm: ; python -m app.llm_generator     # LLM generator çalıştır

uis:
	PYTHONPATH=. streamlit run app/ui_streamlit.py


lst:                                   # modül listesini göster
	@ls app/*.py 2>/dev/null \
	| sed 's#.*/##;s/\.py$$//' \
	| grep -v '^__init__$$' || true
