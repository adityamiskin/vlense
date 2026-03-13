# Release Notes

## 0.2.0

- Added multimodal PDF and image question-answering with page-level retrieval.
- Switched retrieval to ColFlor for smaller document-focused page embeddings.
- Added persistent local collection storage for rendered page images and retrieval embeddings.
- Added a runnable example CLI in `examples/pdf_qa.py`.
- Updated the answer flow to work with OpenAI vision-capable models such as `openai/gpt-5-mini` and `openai/gpt-5.4`.
- Added repository instructions for `uv`-based workflows.
