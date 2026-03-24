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
    model: str = "gpt-5-mini"
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    batch_size: int = 3
    format: Optional[str] = "markdown"
    clean_temp_files: bool = True


@dataclass
class IndexedPage:
    """
    Stored page metadata used for visual retrieval.
    """

    document_id: str
    source_path: str
    file_name: str
    page_number: int
    image_path: str


@dataclass
class RetrievalResult:
    """
    A retrieved page and its similarity score.
    """

    page: IndexedPage
    score: float
