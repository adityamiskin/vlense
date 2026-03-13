from __future__ import annotations

from pathlib import Path
from typing import List, Optional


class ColFlorRetriever:
    """
    Direct ColFlor wrapper for page-image retrieval.
    """

    def __init__(
        self,
        model_name: str = "ahmed-masry/ColFlor",
        device_preference: str = "auto",
    ):
        try:
            import torch
            from PIL import Image
            from colpali_engine.models import ColFlor, ColFlorProcessor
            from colpali_engine.utils.torch_utils import get_torch_device
        except ImportError as exc:
            raise ImportError(
                "ColFlor dependencies are missing. Install project dependencies with `uv sync`."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.device = get_torch_device(device_preference)

        # Upstream ColFlor currently misses newer Transformers capability flags.
        # Force eager attention and disable optional attention backends.
        if not hasattr(ColFlor, "_supports_sdpa"):
            ColFlor._supports_sdpa = False
        if not hasattr(ColFlor, "_supports_flex_attn"):
            ColFlor._supports_flex_attn = False
        if not hasattr(ColFlor, "_supports_flash_attn_2"):
            ColFlor._supports_flash_attn_2 = False

        self.model = ColFlor.from_pretrained(
            model_name,
            device_map=self.device,
            attn_implementation="eager",
        ).eval()
        self.processor = ColFlorProcessor.from_pretrained(model_name)

    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 2,
    ) -> List:
        """
        Encode page images into ColFlor multi-vector embeddings.
        """
        embeddings = []
        for batch_paths in self._batched(image_paths, batch_size):
            images = []
            try:
                for image_path in batch_paths:
                    image = self._image_cls.open(image_path).convert("RGB")
                    images.append(image)

                batch = self.processor.process_images(images)
                batch = self._move_batch_to_device(batch)
                with self._torch.no_grad():
                    batch_embeddings = self.model(**batch)
                embeddings.extend(list(self._torch.unbind(batch_embeddings.to("cpu"))))
            finally:
                for image in images:
                    image.close()
        return embeddings

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 4,
    ) -> List:
        """
        Encode text queries into ColFlor multi-vector embeddings.
        """
        embeddings = []
        for batch_queries in self._batched(queries, batch_size):
            batch = self.processor.process_queries(batch_queries)
            batch = self._move_batch_to_device(batch)
            with self._torch.no_grad():
                batch_embeddings = self.model(**batch)
            embeddings.extend(list(self._torch.unbind(batch_embeddings.to("cpu"))))
        return embeddings

    def score(self, query_embeddings: List, document_embeddings: List):
        """
        Score queries against page embeddings using ColFlor late interaction.
        """
        return self.processor.score(query_embeddings, document_embeddings)

    def save_embeddings(self, embeddings: List, output_path: str) -> None:
        self._torch.save(embeddings, output_path)

    def load_embeddings(self, input_path: str) -> List:
        return self._torch.load(input_path, map_location="cpu")

    def _move_batch_to_device(self, batch: dict):
        return {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    @staticmethod
    def _batched(items: List, batch_size: int):
        for index in range(0, len(items), batch_size):
            yield items[index : index + batch_size]
