# Vlense

Vision-language OCR and multimodal document QA for images and PDFs.

Vlense helps you do two things well:

- extract structured or free-form content from images and PDFs with vision models
- build a page-level retrieval index over documents and ask grounded questions with citations

It is designed for workflows where plain OCR is not enough and the model needs to reason over full document pages, scans, tables, forms, and mixed visual layouts.

## What It Does

- OCR for images and PDFs with Markdown, HTML, or JSON output
- Pydantic schema support for structured extraction
- Page-image indexing for PDFs and image collections
- Multimodal retrieval with `colpali-engine`
- Grounded question answering over retrieved document pages
- Async Python API with a small surface area

## Installation

Install the package:

```bash
uv add vlense
```

Or install from source in this repository:

```bash
uv sync
```

PDF rendering uses `pdf2image`, so Poppler must be available on your system.

## Quick Start

### OCR

```python
import asyncio
import os

from vlense import Vlense


async def main():
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

    vlense = Vlense()
    result = await vlense.ocr(
        file_path=["./invoice.png", "./report.pdf"],
        model="gpt-5-mini",
        format="markdown",
    )

    print(result["invoice.png"].content)


if __name__ == "__main__":
    asyncio.run(main())
```

### Document QA

```python
import asyncio
import os

from vlense import Vlense


async def main():
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

    vlense = Vlense()

    await vlense.index(
        data_dir="./handbook.pdf",
        collection_name="company-docs",
        index_dir="./.vlense",
        retriever_model="vidore/colSmol-500M",
    )

    answer = await vlense.ask(
        query="What are the eligibility requirements?",
        collection_name="company-docs",
        index_dir="./.vlense",
        model="gpt-5-mini",
        top_k=3,
    )

    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

`Vlense.ask()` returns a grounded answer based on the retrieved page images, with cited page references.

For OpenAI-compatible gateways, set `OPENAI_BASE_URL` or pass `base_url=` directly to `Vlense.ocr()` and `Vlense.ask()`.

## Retrieval Model

Vlense uses `colpali-engine` for page-image retrieval and defaults to `vidore/colSmol-500M`.

This gives you:

- document-aware visual retrieval instead of plain text-only chunking
- a smaller default retriever than the heavier ColQwen variants
- a local collection format that stores rendered pages plus embeddings for reuse

## Example CLI

The repository includes a runnable example for PDF question answering:

```bash
uv run python examples/pdf_qa.py ./document.pdf \
  --collection my-docs \
  --question "What does the report say about pricing?" \
  --vision-model gpt-5-mini
```

## API Overview

### `Vlense.ocr()`

Runs OCR over one or more images or PDFs and returns generated content in Markdown, HTML, or JSON.

Key options:

- `file_path`: single path or list of paths
- `model`: OpenAI-compatible vision-capable model name
- `format`: `markdown`, `html`, or `json`
- `json_schema`: optional Pydantic schema for structured extraction
- `output_dir`: optional directory for persisted outputs
- `api_key`: optional API key override
- `base_url`: optional OpenAI-compatible base URL override

### `Vlense.index()`

Builds a local multimodal retrieval collection from PDFs or images.

Key options:

- `data_dir`: file path, list of paths, or directory
- `collection_name`: logical name for the collection
- `index_dir`: storage root for page renders and embeddings
- `retriever_model`: `colpali-engine` checkpoint name

### `Vlense.ask()`

Searches an indexed collection, retrieves the most relevant pages, and asks a vision model to answer using those pages as evidence.

Key options:

- `query`: user question
- `collection_name`: existing indexed collection
- `model`: answer model such as `gpt-5-mini`
- `top_k`: number of retrieved pages to ground the answer
- `api_key`: optional API key override
- `base_url`: optional OpenAI-compatible base URL override

## Release Workflow

GitHub Actions runs CI on pushes and pull requests. Tagged releases publish to PyPI and create a GitHub Release.

Repository setup:

- add a repository secret named `PYPI_API_TOKEN`

Release flow:

```bash
git tag v0.2.5
git push origin v0.2.5
```

## Development

This repository uses `uv`, not `pip`.

Useful commands:

```bash
uv sync
uv run python -m unittest vlense.tests.test_vlense
uv build
```

## Contributing

Issues and pull requests are welcome.

## License

MIT
