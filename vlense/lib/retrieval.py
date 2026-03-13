import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Sequence, Union

from ..core.types import IndexedPage, RetrievalResult

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS
MANIFEST_FILE_NAME = "manifest.json"
EMBEDDINGS_FILE_NAME = "embeddings.pt"


async def pdf_to_images(pdf_path: str, temp_dir: str | None = None):
    """
    Lazy wrapper so importing retrieval does not require pdf/OCR dependencies.
    """
    from .pdf import pdf_to_images as pdf_to_images_impl

    return await pdf_to_images_impl(pdf_path, temp_dir)


def resolve_input_files(file_input: Union[str, Sequence[str]]) -> List[str]:
    """
    Resolve one or more file paths or directories into supported document paths.
    """
    if isinstance(file_input, str):
        candidates = [file_input]
    else:
        candidates = list(file_input)

    resolved: List[str] = []
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and file_path.is_file():
                    resolved.append(str(file_path))
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        resolved.append(str(path))

    if not resolved:
        raise ValueError("No supported files found to index.")

    return resolved


def get_collection_path(index_dir: str, collection_name: str) -> Path:
    """
    Collection directories are stored under the chosen index root.
    """
    return Path(index_dir).expanduser().resolve() / collection_name


def get_manifest_path(index_dir: str, collection_name: str) -> Path:
    """
    Return the manifest location for a collection.
    """
    return get_collection_path(index_dir, collection_name) / MANIFEST_FILE_NAME


def get_embeddings_path(index_dir: str, collection_name: str) -> Path:
    """
    Return the serialized embeddings location for a collection.
    """
    return get_collection_path(index_dir, collection_name) / EMBEDDINGS_FILE_NAME


def build_document_id(source_path: str) -> str:
    """
    Generate a stable ID for a source document path.
    """
    return sha256(str(Path(source_path).resolve()).encode("utf-8")).hexdigest()[:16]


async def materialize_document_pages(
    source_path: str,
    pages_root: Path,
    temp_directory: str | None = None,
) -> List[Dict[str, Union[int, str]]]:
    """
    Convert a source document into stored page images under the collection.
    """
    source = Path(source_path).resolve()
    document_id = build_document_id(str(source))
    document_root = pages_root / f"{source.stem}-{document_id}"
    document_root.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() in SUPPORTED_PDF_EXTENSIONS:
        rendered_images = await pdf_to_images(str(source), temp_directory)
        if not rendered_images:
            raise ValueError(f"Failed to convert PDF into page images: {source}")
        page_entries = []
        for page_number, rendered_image in enumerate(rendered_images, start=1):
            destination = document_root / f"page-{page_number:04d}.png"
            shutil.copy2(rendered_image, destination)
            page_entries.append(
                {
                    "document_id": document_id,
                    "source_path": str(source),
                    "file_name": source.name,
                    "page_number": page_number,
                    "image_path": str(destination),
                }
            )
        return page_entries

    destination = document_root / f"page-0001{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return [
        {
            "document_id": document_id,
            "source_path": str(source),
            "file_name": source.name,
            "page_number": 1,
            "image_path": str(destination),
        }
    ]


def save_manifest(
    index_dir: str,
    collection_name: str,
    retriever_model: str,
    pages: List[IndexedPage],
) -> str:
    """
    Persist the collection manifest to disk.
    """
    collection_path = get_collection_path(index_dir, collection_name)
    collection_path.mkdir(parents=True, exist_ok=True)
    manifest_path = collection_path / MANIFEST_FILE_NAME

    payload = {
        "collection_name": collection_name,
        "retriever": "colpali",
        "retriever_model": retriever_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": [asdict(page) for page in pages],
        "embeddings_path": str(get_embeddings_path(index_dir, collection_name)),
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(manifest_path)


def load_manifest(index_dir: str, collection_name: str) -> Dict:
    """
    Load a collection manifest from disk.
    """
    manifest_path = get_manifest_path(index_dir, collection_name)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Collection '{collection_name}' was not found under {manifest_path.parent}."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def parse_indexed_pages(payload: Dict) -> List[IndexedPage]:
    """
    Hydrate stored manifest data into dataclasses.
    """
    return [IndexedPage(**page) for page in payload.get("pages", [])]


def rank_pages(
    pages: List[IndexedPage],
    scores,
    top_k: int,
) -> List[RetrievalResult]:
    """
    Rank pages with ColPali scores and return top results.
    """
    if hasattr(scores, "tolist"):
        score_values = scores.tolist()
    else:
        score_values = list(scores)

    if score_values and isinstance(score_values[0], list):
        score_values = score_values[0]

    ranked = [
        RetrievalResult(page=page, score=float(score))
        for page, score in zip(pages, score_values)
    ]
    ranked.sort(key=lambda result: result.score, reverse=True)
    return ranked[:top_k]
