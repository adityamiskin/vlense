import os
import shutil
import aioshutil
import aiofiles.os as async_os
import tempfile
from typing import Optional, Union, List, Dict, Type, Any

from .types import VlenseResponse
from ..lib import process_batch_with_completion
from ..models import LiteLLMModel

from pydantic import BaseModel  # Import BaseModel for type hinting


class Vlense:
    """
    Vlense class for performing OCR and handling document queries.
    """

    def __init__(self):
        pass

    async def ocr(
        self,
        file_path: Union[str, List[str]],
        model: str = "gemini-1.5-flash",
        output_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        batch_size: int = 3,
        format: str = "markdown",
        json_schema: Optional[Type[BaseModel]] = None,  # New parameter
        clean_temp_files: Optional[bool] = True,
    ) -> Dict[str, VlenseResponse]:
        """
        API to perform OCR to markdown, html, or structured JSON using Vision models.
        Supports both PDF and image files.
        Please set up the environment variables for the model and model provider before using this API. Refer: https://docs.litellm.ai/docs/providers

        :param file_path: The path or URL to the PDF/image file to process.
        :type file_path: Union[str, List[str]]
        :param model: The model to use for generating completions, defaults to "gemini-1.5-flash". Note - Refer: https://docs.litellm.ai/docs/providers to pass the correct model name as per provider specifications.
        :type model: str, optional
        :param output_dir: The directory to save the output, defaults to None.
        :type output_dir: Optional[str], optional
        :param temp_dir: The directory to store temporary files, defaults to a named folder in the system's temp directory. If it exists, its contents will be deleted.
        :type temp_dir: Optional[str], optional
        :param batch_size: The number of concurrent processes to run, defaults to 3.
        :type batch_size: int, optional
        :param format: Output format, either 'markdown', 'html', or 'json'.
        :type format: str, optional
        :param json_schema: Pydantic model defining the JSON schema for structured output, required if format is 'json'.
        :type json_schema: Optional[Type[BaseModel]], optional
        :param clean_temp_files: Whether to cleanup the temporary files after processing, defaults to True.
        :type clean_temp_files: Optional[bool], optional
        :return: The content generated by the model.
        :rtype: Dict[str, VlenseResponse]
        """

        if format not in ["markdown", "html", "json"]:
            raise ValueError("Format must be either 'markdown', 'html', or 'json'")

        if format == "json" and not json_schema:
            raise ValueError("json_schema must be provided when format is 'json'")

        results = {}

        if isinstance(file_path, str):
            file_paths = [file_path]
        else:
            file_paths = file_path

        if output_dir:
            await async_os.makedirs(output_dir, exist_ok=True)

        if temp_dir:
            if os.path.exists(temp_dir):
                await aioshutil.rmtree(temp_dir)
            await async_os.makedirs(temp_dir, exist_ok=True)

        llm_model = LiteLLMModel(model=model, format=format, json_schema=json_schema)

        with tempfile.TemporaryDirectory() as temp_dir_:
            if temp_dir:
                temp_directory = temp_dir
            else:
                temp_directory = temp_dir_

            # Process all files using the updated process_batch_with_completion
            batch_results = await process_batch_with_completion(
                model=llm_model,
                file_paths=file_paths,
                batch_size=batch_size,
                format=format,
                output_dir=output_dir,
                temp_directory=temp_directory,
            )

            results.update(batch_results)

            if clean_temp_files and temp_dir:
                shutil.rmtree(temp_dir)

        return results

    async def index(
        self,
        data_dir: str,
        collection_name: str,
    ) -> str:
        """
        Index the provided document for future queries.

        Args:
            file_path (str): The path to the document to be indexed.
            model (str): The model to use for indexing the document.

        Returns:
            str: The index of the document.
        """
        # Placeholder for future implementation
        return "This feature will be implemented soon."

    async def ask(
        self,
        query: str,
        collection_name: str,
        model: str = "gemini-1.5-flash",
    ) -> str:
        """
        Answer questions based on the provided documents.

        Args:
            query (str): The question to be answered.
            context (Union[str, List[str]]): Optional context from documents to base the answer on.
            model (str): The model to use for generating the answer.

        Returns:
            str: The answer to the query.
        """
        # Placeholder for future implementation
        return "This feature will be implemented soon."
