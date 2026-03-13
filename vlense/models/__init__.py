__all__ = ["LiteLLMModel", "ColPaliRetriever"]


def __getattr__(name):
    if name == "LiteLLMModel":
        from .litellmmodel import LiteLLMModel

        return LiteLLMModel
    if name == "ColPaliRetriever":
        from .colpali import ColPaliRetriever

        return ColPaliRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
