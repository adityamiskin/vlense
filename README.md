# Vlense

A Python package to extract text from images and PDFs using Vision Language Models (VLM).

## Features

- Extract text from images and PDFs
- Supports JSON, HTML, and Markdown formats
- Easy integration with Vision Language Models
- Asynchronous processing with batch support
- Custom JSON schema for structured output
- Build page-image embedding collections for multimodal RAG
- Ask grounded questions over indexed PDFs and images

## Installation

```bash
uv sync
```

## Usage

```python
import os
import asyncio
from vlense import Vlense
from pydantic import BaseModel

path = ["./images/image1.jpg", "test.pdf"]
output_dir = "./output"
model = "openai/gpt-5-mini"
temp_dir = "./temp_images"
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"


async def main():
    vlense = Vlense()
    responses = await vlense.ocr(
        file_path=path,
        model=model,
        output_dir=output_dir,
        temp_dir=temp_dir,
        batch_size=3,
        clean_temp_files=False,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Multimodal RAG

`Vlense.index()` builds a `colpali-engine` retrieval index for PDFs and images. `Vlense.ask()` searches that index for the most relevant document pages and sends the retrieved page images to the vision model to answer with citations.

```python
import asyncio
from vlense import Vlense


async def main():
    vlense = Vlense()

    await vlense.index(
        data_dir=["./handbook.pdf", "./diagram.png"],
        collection_name="company-docs",
        index_dir="./.vlense",
        retriever_model="vidore/colSmol-500M",
    )

    answer = await vlense.ask(
        query="What are the eligibility requirements?",
        collection_name="company-docs",
        index_dir="./.vlense",
        model="openai/gpt-5-mini",
        top_k=3,
    )

    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

Retrieval uses `colpali-engine` directly and defaults to `vidore/colSmol-500M`, which is smaller than full ColQwen2-family checkpoints while staying document-focused.

## Publishing

GitHub Actions builds and tests the package on every push and pull request. Publishing to PyPI happens on tags matching `v*` through PyPI trusted publishing.

One-time PyPI setup:

- In PyPI, add a trusted publisher for this GitHub repository.
- Set the workflow name to `publish.yaml`.
- Set the environment name to `pypi`.

Release flow:

```bash
git tag v0.2.2
git push origin v0.2.2
```

## API

### Vlense.ocr()

Performs OCR on the provided files.

**Parameters:**

- file_path : (Union[str, List[str]]): Path or list of paths to PDF/image files.

- model : (str, optional): Model name for generating completions. Defaults to `"gemini-flash-latest"`.

- output_dir : (Optional[str], optional): Directory to save output. Defaults to `None`.

- temp_dir : (Optional[str], optional): Directory for temporary files. Defaults to system temp.

- batch_size : (int, optional): Number of concurrent processes. Defaults to `3`.

- format : (str, optional): Output format (`'markdown'`, `'html'`, `'json'`). Defaults to `'markdown'`.

- json_schema : (Optional[Type[BaseModel]], optional): Pydantic model for JSON output. Required if format is `'json'`.

- clean_temp_files : (Optional[bool], optional): Cleanup temporary files after processing. Defaults to `True`.

**Returns:**

- Dict[str, VlenseResponse] : Generated content.

### Vlense.index()

Indexes one or more PDFs or images into a local page-image retrieval collection.

**Parameters:**

- data_dir : (Union[str, List[str]]): File path, list of file paths, or a directory containing supported PDF/image files.
- collection_name : (str): Name of the collection to create or replace.
- index_dir : (str, optional): Root directory used to store indexed collections. Defaults to `".vlense"`.
- retriever_model : (str, optional): `colpali-engine` model name. Defaults to `"vidore/colSmol-500M"`.
- embedding_batch_size : (int, optional): Batch size used while embedding page images. Defaults to `2`.
- temp_dir : (Optional[str], optional): Temporary directory for rendered PDF pages.

**Returns:**

- str : Path to the collection manifest.

### Vlense.ask()

Answers a question using retrieved page images from a previously indexed collection.

**Parameters:**

- query : (str): User question.
- collection_name : (str): Indexed collection name.
- model : (str, optional): Vision model used for grounded answering. Defaults to `"gemini-flash-latest"`.
- index_dir : (str, optional): Root directory where collections are stored. Defaults to `".vlense"`.
- top_k : (int, optional): Number of retrieved pages to send to the model. Defaults to `3`.

**Returns:**

- str : Grounded answer with cited pages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

Author: Aditya Miskin  
Email: [adityamiskin98@gmail.com](mailto:adityamiskin98@gmail.com)  
Repository: [https://github.com/adityamiskin/vlense](https://github.com/adityamiskin/vlense)
