import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vlense import Vlense

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


class FakeColPaliRetriever:
    last_init = None
    last_encode_images = None
    last_encode_queries = None
    last_score_inputs = None
    saved_embeddings = {}

    def __init__(self, model_name="vidore/colSmol-500M", device_preference="auto"):
        FakeColPaliRetriever.last_init = {
            "model_name": model_name,
            "device_preference": device_preference,
        }

    def encode_images(self, image_paths, batch_size=2):
        FakeColPaliRetriever.last_encode_images = {
            "image_paths": image_paths,
            "batch_size": batch_size,
        }
        return [f"emb:{Path(path).name}" for path in image_paths]

    def encode_queries(self, queries, batch_size=4):
        FakeColPaliRetriever.last_encode_queries = {
            "queries": queries,
            "batch_size": batch_size,
        }
        return ["query-embedding"]

    def score(self, query_embeddings, document_embeddings):
        FakeColPaliRetriever.last_score_inputs = {
            "query_embeddings": query_embeddings,
            "document_embeddings": document_embeddings,
        }
        return [[0.1, 0.9]]

    def save_embeddings(self, embeddings, output_path):
        FakeColPaliRetriever.saved_embeddings[output_path] = embeddings
        Path(output_path).write_text("stub", encoding="utf-8")

    def load_embeddings(self, input_path):
        return FakeColPaliRetriever.saved_embeddings[input_path]


class FakeOpenAIModel:
    last_call = None

    def __init__(self, model, format="markdown", json_schema=None, api_key=None, base_url=None):
        self.model = model
        self.format = format
        self.json_schema = json_schema
        self.api_key = api_key
        self.base_url = base_url

    async def answer_question(self, question, image_paths, page_references, text_context=None):
        FakeOpenAIModel.last_call = {
            "question": question,
            "image_paths": image_paths,
            "page_references": page_references,
            "text_context": text_context,
        }
        return SimpleNamespace(content="Grounded answer")


def write_text_pdf(path: Path, pages: list[str]) -> None:
    if fitz is None:  # pragma: no cover
        raise unittest.SkipTest("PyMuPDF is not installed")

    doc = fitz.open()
    try:
        for page_text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), page_text)
        doc.save(path)
    finally:
        doc.close()


class TestVlenseRag(unittest.IsolatedAsyncioTestCase):
    async def test_index_creates_colpali_manifest_for_pdf(self):
        async def fake_materialize_document_pages(source_path, pages_root, temp_directory=None):
            page_path = pages_root / "report-doc" / "page-0001.png"
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text("img", encoding="utf-8")
            return [
                {
                    "document_id": "doc-1",
                    "source_path": str(Path(source_path).resolve()),
                    "file_name": Path(source_path).name,
                    "page_number": 1,
                    "image_path": str(page_path),
                }
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            fake_colpali_module = ModuleType("vlense.models.colpali")
            fake_colpali_module.ColPaliRetriever = FakeColPaliRetriever

            with patch("vlense.core.vlense.materialize_document_pages", side_effect=fake_materialize_document_pages), patch.dict(
                sys.modules,
                {"vlense.models.colpali": fake_colpali_module},
            ):
                manifest_path = await Vlense().index(
                    data_dir=str(pdf_path),
                    collection_name="reports",
                    index_dir=str(temp_root / "index"),
                    temp_dir=str(temp_root / "scratch"),
                )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

            self.assertEqual(manifest["collection_name"], "reports")
            self.assertEqual(manifest["retriever"], "colpali")
            self.assertEqual(manifest["retriever_model"], "vidore/colSmol-500M")
            self.assertEqual(len(manifest["pages"]), 1)
            self.assertTrue(Path(manifest["embeddings_path"]).exists())
            self.assertEqual(
                FakeColPaliRetriever.last_encode_images["batch_size"],
                2,
            )
            self.assertEqual(
                FakeColPaliRetriever.last_encode_images["image_paths"][0].endswith("page-0001.png"),
                True,
            )

    async def test_ask_uses_colpali_results_before_answering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            index_root = temp_root / "index"
            collection_root = index_root / "reports"
            collection_root.mkdir(parents=True, exist_ok=True)
            image_path = temp_root / "page-0002.png"
            image_path.write_text("img", encoding="utf-8")
            embeddings_path = collection_root / "embeddings.pt"
            FakeColPaliRetriever.saved_embeddings[str(embeddings_path)] = ["emb:page-0001.png", "emb:page-0002.png"]
            manifest = {
                "collection_name": "reports",
                "retriever": "colpali",
                "retriever_model": "vidore/colSmol-500M",
                "embeddings_path": str(embeddings_path),
                "pages": [
                    {
                        "document_id": "doc-1",
                        "source_path": str(temp_root / "report.pdf"),
                        "file_name": "report.pdf",
                        "page_number": 1,
                        "image_path": str(temp_root / "page-0001.png"),
                    },
                    {
                        "document_id": "doc-1",
                        "source_path": str(temp_root / "report.pdf"),
                        "file_name": "report.pdf",
                        "page_number": 2,
                        "image_path": str(image_path),
                    },
                ],
            }
            (collection_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            fake_colpali_module = ModuleType("vlense.models.colpali")
            fake_colpali_module.ColPaliRetriever = FakeColPaliRetriever
            fake_openai_module = ModuleType("vlense.models.openai_model")
            fake_openai_module.OpenAIModel = FakeOpenAIModel

            with patch.dict(
                sys.modules,
                {
                    "vlense.models.colpali": fake_colpali_module,
                    "vlense.models.openai_model": fake_openai_module,
                },
            ):
                answer = await Vlense().ask(
                    query="What does page two say?",
                    collection_name="reports",
                    index_dir=str(index_root),
                    top_k=1,
                )

            self.assertEqual(answer, "Grounded answer")
            self.assertEqual(FakeColPaliRetriever.last_encode_queries["queries"], ["What does page two say?"])
            self.assertEqual(
                FakeColPaliRetriever.last_score_inputs["document_embeddings"],
                ["emb:page-0001.png", "emb:page-0002.png"],
            )
            self.assertEqual(FakeOpenAIModel.last_call["image_paths"], [str(image_path)])
            self.assertEqual(
                FakeOpenAIModel.last_call["page_references"],
                ["report.pdf p.2"],
            )

    async def test_index_creates_bm25_manifest_for_pdf(self):
        async def fake_materialize_document_pages(source_path, pages_root, temp_directory=None):
            rendered_paths = []
            for page_number in range(1, 3):
                page_path = pages_root / "report-doc" / f"page-{page_number:04d}.png"
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(f"img-{page_number}", encoding="utf-8")
                rendered_paths.append(
                    {
                        "document_id": "doc-1",
                        "source_path": str(Path(source_path).resolve()),
                        "file_name": Path(source_path).name,
                        "page_number": page_number,
                        "image_path": str(page_path),
                    }
                )
            return rendered_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "report.pdf"
            write_text_pdf(
                pdf_path,
                [
                    "Alpha section discusses apples and pears.",
                    "Beta section discusses bananas and grapes.",
                ],
            )

            with patch("vlense.core.vlense.materialize_document_pages", side_effect=fake_materialize_document_pages):
                manifest_path = await Vlense().index(
                    data_dir=str(pdf_path),
                    collection_name="reports",
                    index_dir=str(temp_root / "index"),
                    retrieval="bm25",
                    temp_dir=str(temp_root / "scratch"),
                )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

            self.assertEqual(manifest["retriever"], "bm25")
            self.assertEqual(len(manifest["pages"]), 2)
            self.assertGreaterEqual(len(manifest["chunks"]), 1)
            self.assertTrue(
                any("bananas" in chunk["text"].lower() for chunk in manifest["chunks"])
            )

    async def test_ask_uses_bm25_results_before_answering(self):
        async def fake_materialize_document_pages(source_path, pages_root, temp_directory=None):
            rendered_paths = []
            for page_number in range(1, 3):
                page_path = pages_root / "report-doc" / f"page-{page_number:04d}.png"
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(f"img-{page_number}", encoding="utf-8")
                rendered_paths.append(
                    {
                        "document_id": "doc-1",
                        "source_path": str(Path(source_path).resolve()),
                        "file_name": Path(source_path).name,
                        "page_number": page_number,
                        "image_path": str(page_path),
                    }
                )
            return rendered_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pdf_path = temp_root / "report.pdf"
            write_text_pdf(
                pdf_path,
                [
                    "Alpha page covers apples and pears.",
                    "Banana policy is described here in detail.",
                ],
            )

            with patch("vlense.core.vlense.materialize_document_pages", side_effect=fake_materialize_document_pages):
                await Vlense().index(
                    data_dir=str(pdf_path),
                    collection_name="reports",
                    index_dir=str(temp_root / "index"),
                    retrieval="bm25",
                    temp_dir=str(temp_root / "scratch"),
                )

            fake_openai_module = ModuleType("vlense.models.openai_model")
            fake_openai_module.OpenAIModel = FakeOpenAIModel

            with patch.dict(sys.modules, {"vlense.models.openai_model": fake_openai_module}):
                answer = await Vlense().ask(
                    query="What does the document say about banana policy?",
                    collection_name="reports",
                    index_dir=str(temp_root / "index"),
                    retrieval="bm25",
                    top_k=1,
                )

            self.assertEqual(answer, "Grounded answer")
            self.assertEqual(len(FakeOpenAIModel.last_call["image_paths"]), 1)
            self.assertEqual(
                FakeOpenAIModel.last_call["page_references"],
                ["report.pdf p.2"],
            )
            self.assertIn("Banana policy", FakeOpenAIModel.last_call["text_context"])


if __name__ == "__main__":
    unittest.main()
