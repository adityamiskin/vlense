from __future__ import annotations

from typing import List


class ColPaliRetriever:
    """
    Direct colpali-engine wrapper for page-image retrieval.

    The same ColQwen2 model/processor classes are used for ColSmol checkpoints.
    """

    def __init__(
        self,
        model_name: str = "vidore/colSmol-500M",
        device_preference: str = "auto",
    ):
        try:
            import torch
            from PIL import Image
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            from colpali_engine.utils.torch_utils import get_torch_device
        except ImportError as exc:
            raise ImportError(
                "colpali-engine dependencies are missing. Install project dependencies with `uv sync`."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.device = get_torch_device(device_preference)
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 2,
    ) -> List:
        """
        Encode page images into multi-vector embeddings.
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
        Encode text queries into multi-vector embeddings.
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
        Score queries against page embeddings using late interaction.
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
