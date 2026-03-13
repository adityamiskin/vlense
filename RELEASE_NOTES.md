# Release Notes

## 0.2.3

- Switched the PyPI publish workflow to use a repository `PYPI_API_TOKEN` secret instead of PyPI trusted publishing.
- Kept tagged releases creating both the PyPI upload and the GitHub Release entry.

## 0.2.2

- Fixed `vidore/colSmol-500M` loading by selecting the correct `colpali-engine` backend from checkpoint metadata.
- Added a `uv`-based GitHub Actions pipeline that tests, builds, and publishes tagged releases to PyPI with trusted publishing.

## 0.2.1

- Replaced the non-publishable Git dependency with the PyPI `colpali-engine` package.
- Switched the default retriever to `vidore/colSmol-500M` through direct `colpali-engine` integration.
- Kept the multimodal PDF QA flow while making the package publishable on PyPI.

## 0.2.0

- Added multimodal PDF and image question-answering with page-level retrieval.
- Switched retrieval to direct `colpali-engine` integration with smaller document-focused page embeddings.
- Added persistent local collection storage for rendered page images and retrieval embeddings.
- Added a runnable example CLI in `examples/pdf_qa.py`.
- Updated the answer flow to work with OpenAI vision-capable models such as `openai/gpt-5-mini` and `openai/gpt-5.4`.
- Added repository instructions for `uv`-based workflows.
