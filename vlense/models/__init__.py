__all__ = ["LiteLLMModel", "ColFlorRetriever"]


def __getattr__(name):
    if name == "LiteLLMModel":
        from .litellmmodel import LiteLLMModel

        return LiteLLMModel
    if name == "ColFlorRetriever":
        from .colflor import ColFlorRetriever

        return ColFlorRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
