# Vlense

A Python package to extract text from images and PDFs using Vision Language Models (VLM).

## Features

- Extract text from images and PDFs
- Supports JSON, HTML, and Markdown formats
- Easy integration with Vision Language Models
- Asynchronous processing with batch support
- Custom JSON schema for structured output

## Installation

```bash
pip install vlense
```

## Usage

```python
import os
import asyncio
from vlense import Vlense
from pydantic import BaseModel

path = ["./images/image1.jpg", "test.pdf"]
output_dir = "./output"
model = "gemini/gemini-1.5-flash"
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

## API

### Vlense.ocr()

Performs OCR on the provided files.

**Parameters:**

- file_path : (Union[str, List[str]]): Path or list of paths to PDF/image files.

- model : (str, optional): Model name for generating completions. Defaults to `"gemini-1.5-flash"`.

- output_dir : (Optional[str], optional): Directory to save output. Defaults to `None`.

- temp_dir : (Optional[str], optional): Directory for temporary files. Defaults to system temp.

- batch_size : (int, optional): Number of concurrent processes. Defaults to `3`.

- format : (str, optional): Output format (`'markdown'`, `'html'`, `'json'`). Defaults to `'markdown'`.

- json_schema : (Optional[Type[BaseModel]], optional): Pydantic model for JSON output. Required if format is `'json'`.

- clean_temp_files : (Optional[bool], optional): Cleanup temporary files after processing. Defaults to `True`.

**Returns:**

- Dict[str, VlenseResponse] : Generated content.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

Author: Aditya Miskin  
Email: [adityamiskin98@gmail.com](mailto:adityamiskin98@gmail.com)  
Repository: [https://github.com/adityamiskin/vlense](https://github.com/adityamiskin/vlense)
