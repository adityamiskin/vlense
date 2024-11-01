from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Page:
    """
    Dataclass to store the page content.
    """

    content: str
    content_length: int
    input_tokens: int
    output_tokens: int
    page: int


@dataclass
class VlenseResponse:
    """
    A class representing the response of a completion.
    """

    completion_time: float
    file_name: str
    total_input_tokens: int
    total_output_tokens: int
    pages: List[Page]


@dataclass
class VlenseArgs:
    """
    Dataclass to store the arguments for the Vlense class.
    """

    file_path: Union[str, List[str]]
    model: str = "gemini-1.5-flash-8b"
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    batch_size: int = 3
    format: Optional[str] = "markdown"
    clean_temp_files: bool = True
