import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vlense import Vlense


class FakeColFlorRetriever:
    last_init = None
    last_encode_images = None
    last_encode_queries = None
    last_score_inputs = None
    saved_embeddings = {}

    def __init__(self, model_name="ahmed-masry/ColFlor", device_preference="auto"):
        FakeColFlorRetriever.last_init = {
            "model_name": model_name,
            "device_preference": device_preference,
        }

    def encode_images(self, image_paths, batch_size=2):
        FakeColFlorRetriever.last_encode_images = {
            "image_paths": image_paths,
            "batch_size": batch_size,
        }
        return [f"emb:{Path(path).name}" for path in image_paths]

    def encode_queries(self, queries, batch_size=4):
        FakeColFlorRetriever.last_encode_queries = {
            "queries": queries,
            "batch_size": batch_size,
        }
        return ["query-embedding"]

    def score(self, query_embeddings, document_embeddings):
        FakeColFlorRetriever.last_score_inputs = {
            "query_embeddings": query_embeddings,
            "document_embeddings": document_embeddings,
        }
        return [[0.1, 0.9]]

    def save_embeddings(self, embeddings, output_path):
        FakeColFlorRetriever.saved_embeddings[output_path] = embeddings
        Path(output_path).write_text("stub", encoding="utf-8")

    def load_embeddings(self, input_path):
        return FakeColFlorRetriever.saved_embeddings[input_path]


class FakeLiteLLMModel:
    last_call = None

    def __init__(self, model, format="markdown", json_schema=None):
        self.model = model
        self.format = format
        self.json_schema = json_schema

    async def answer_question(self, question, image_paths, page_references):
        FakeLiteLLMModel.last_call = {
            "question": question,
            "image_paths": image_paths,
            "page_references": page_references,
        }
        return SimpleNamespace(content="Grounded answer")


class TestVlenseRag(unittest.IsolatedAsyncioTestCase):
    async def test_index_creates_colflor_manifest_for_pdf(self):
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

            fake_colflor_module = ModuleType("vlense.models.colflor")
            fake_colflor_module.ColFlorRetriever = FakeColFlorRetriever

            with patch("vlense.core.vlense.materialize_document_pages", side_effect=fake_materialize_document_pages), patch.dict(
                sys.modules,
                {"vlense.models.colflor": fake_colflor_module},
            ):
                manifest_path = await Vlense().index(
                    data_dir=str(pdf_path),
                    collection_name="reports",
                    index_dir=str(temp_root / "index"),
                    temp_dir=str(temp_root / "scratch"),
                )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

            self.assertEqual(manifest["collection_name"], "reports")
            self.assertEqual(manifest["retriever"], "colflor")
            self.assertEqual(manifest["retriever_model"], "ahmed-masry/ColFlor")
            self.assertEqual(len(manifest["pages"]), 1)
            self.assertTrue(Path(manifest["embeddings_path"]).exists())
            self.assertEqual(
                FakeColFlorRetriever.last_encode_images["batch_size"],
                2,
            )
            self.assertEqual(
                FakeColFlorRetriever.last_encode_images["image_paths"][0].endswith("page-0001.png"),
                True,
            )

    async def test_ask_uses_colflor_results_before_answering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            index_root = temp_root / "index"
            collection_root = index_root / "reports"
            collection_root.mkdir(parents=True, exist_ok=True)
            image_path = temp_root / "page-0002.png"
            image_path.write_text("img", encoding="utf-8")
            embeddings_path = collection_root / "embeddings.pt"
            FakeColFlorRetriever.saved_embeddings[str(embeddings_path)] = ["emb:page-0001.png", "emb:page-0002.png"]
            manifest = {
                "collection_name": "reports",
                "retriever": "colflor",
                "retriever_model": "ahmed-masry/ColFlor",
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

            fake_colflor_module = ModuleType("vlense.models.colflor")
            fake_colflor_module.ColFlorRetriever = FakeColFlorRetriever
            fake_litellm_module = ModuleType("vlense.models.litellmmodel")
            fake_litellm_module.LiteLLMModel = FakeLiteLLMModel

            with patch.dict(
                sys.modules,
                {
                    "vlense.models.colflor": fake_colflor_module,
                    "vlense.models.litellmmodel": fake_litellm_module,
                },
            ):
                answer = await Vlense().ask(
                    query="What does page two say?",
                    collection_name="reports",
                    index_dir=str(index_root),
                    top_k=1,
                )

            self.assertEqual(answer, "Grounded answer")
            self.assertEqual(FakeColFlorRetriever.last_encode_queries["queries"], ["What does page two say?"])
            self.assertEqual(
                FakeColFlorRetriever.last_score_inputs["document_embeddings"],
                ["emb:page-0001.png", "emb:page-0002.png"],
            )
            self.assertEqual(FakeLiteLLMModel.last_call["image_paths"], [str(image_path)])
            self.assertEqual(
                FakeLiteLLMModel.last_call["page_references"],
                ["report.pdf p.2"],
            )


if __name__ == "__main__":
    unittest.main()
