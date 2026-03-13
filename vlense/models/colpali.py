from __future__ import annotations

from importlib import import_module
from typing import List


class ColPaliRetriever:
    """
    Direct colpali-engine wrapper for page-image retrieval.
    """

    def __init__(
        self,
        model_name: str = "vidore/colSmol-500M",
        device_preference: str = "auto",
    ):
        try:
            import torch
            from PIL import Image
            from colpali_engine.utils.torch_utils import get_torch_device
            from transformers import AutoConfig
        except ImportError as exc:
            raise ImportError(
                "colpali-engine dependencies are missing. Install project dependencies with `uv sync`."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.device = get_torch_device(device_preference)
        model_cls, processor_cls = self._resolve_components(
            config=AutoConfig.from_pretrained(model_name)
        )
        self.model = model_cls.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.processor = processor_cls.from_pretrained(model_name)

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

    @staticmethod
    def _resolve_components(config):
        models = import_module("colpali_engine.models")
        architecture = next(iter(getattr(config, "architectures", [])), "")
        model_type = getattr(config, "model_type", "")

        if architecture == "ColIdefics3" or model_type == "idefics3":
            return models.ColIdefics3, models.ColIdefics3Processor

        if architecture == "ColQwen2_5" or model_type == "qwen2_5_vl":
            return models.ColQwen2_5, models.ColQwen2_5_Processor

        if architecture == "ColQwen2" or model_type == "qwen2_vl":
            return models.ColQwen2, models.ColQwen2Processor

        if architecture == "ColPali" or model_type == "paligemma":
            return models.ColPali, models.ColPaliProcessor

        raise ValueError(
            f"Unsupported colpali-engine checkpoint architecture {architecture!r} "
            f"for model_type {model_type!r}."
        )
