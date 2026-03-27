# Release Notes

## 0.2.6

- Added `bm25` and `hybrid` retrieval backends alongside the existing ColPali page-image retrieval flow.
- Added text-layer PDF chunking and lexical retrieval with `PyMuPDF` and `rank-bm25`.
- Updated grounded QA prompts and answer assembly so questions can use retrieved text excerpts together with page images.
- Extended tests and README examples to cover the new retrieval options.

## 0.2.5

- Removed the LiteLLM dependency and all LiteLLM-specific code paths.
- Replaced the vision model adapter with a direct OpenAI-compatible client using `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- Updated tests, examples, and documentation to match the new runtime configuration.

## 0.2.4

- Rewrote the README and package metadata with a clearer professional description of the OCR and multimodal document QA workflow.
- Updated the published package summary to better reflect page-level retrieval and grounded question answering.

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
- Updated the answer flow to work with OpenAI vision-capable models such as `gpt-5-mini` and `gpt-5.4`.
- Added repository instructions for `uv`-based workflows.
