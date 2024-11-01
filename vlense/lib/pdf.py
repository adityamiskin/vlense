import time
from typing import List, Optional, Dict
import asyncio
import os
from pdf2image import convert_from_path
import shutil
from pathlib import Path

import aiofiles

from .text import extract_html_content, format_markdown, get_final_html


from ..core import VlenseResponse, Page
from ..models import LiteLLMModel


async def pdf_to_images(pdf_path: str, temp_dir: str = None) -> List[str]:
    """Convert a PDF file to images asynchronously using pdf2image's threading."""

    options = {
        "pdf_path": pdf_path,
        "output_folder": temp_dir,
        "fmt": "png",
        "thread_count": 4,
        "use_pdftocairo": True,
        "paths_only": True,
    }

    try:
        images = await asyncio.to_thread(convert_from_path, **options)
        return images

    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


async def process_batch_with_completion(
    model,
    file_paths: List[str],
    batch_size=3,
    format="markdown",
    output_dir: Optional[str] = None,
    temp_directory: Optional[str] = None,
) -> Dict[str, VlenseResponse]:
    """Process files in batches to control concurrency and return VlenseResponse objects."""
    results = {}

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        batch_tasks = []
        for path in batch:
            batch_tasks.append(
                process_file(path, model, format, output_dir, temp_directory)
            )
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        for path, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                print(f"Error processing {path}: {result}")
            else:
                results[path] = result
    return results


async def process_file(
    path: str,
    llm_model: LiteLLMModel,
    format: str,
    output_dir: Optional[str],
    temp_directory: str,
) -> VlenseResponse:
    """Process a single PDF or image file and return a VlenseResponse."""
    start_time = time.perf_counter()
    file_ext = Path(path).suffix.lower()

    if file_ext in [".pdf"]:
        image_paths = await pdf_to_images(path, temp_directory)
    elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        destination = Path(temp_directory) / Path(path).name
        shutil.copy(path, destination)
        image_paths = [str(destination)]
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    responses = []
    for image_path in image_paths:
        response = await llm_model.completion(image_path)
        responses.append(response)

    pages = []
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, response in enumerate(responses):
        if response is None:
            raise AttributeError(f"Response for {path} is None")
        page_content = response.content
        pages.append(
            Page(
                content=page_content,
                content_length=len(page_content),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                page=idx + 1,
            )
        )
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

    combined_content = "\n\n".join(page.content for page in pages)
    if format == "html":
        html_content = extract_html_content(combined_content)
        combined_content = get_final_html(html_content)
    elif format == "json":
        combined_content = combined_content  # Already JSON formatted
    else:
        combined_content = format_markdown(combined_content)

    if output_dir:
        base_name = os.path.splitext(os.path.basename(path))[0]
        if format == "json":
            extension = ".json"
        elif format == "html":
            extension = ".html"
        else:
            extension = ".md"
        output_path = os.path.join(output_dir, f"{base_name}{extension}")

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(combined_content)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return VlenseResponse(
        file_name=os.path.basename(path),
        completion_time=elapsed_time,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        pages=pages,
        # total_content=combined_content,
    )
