__all__ = ["OpenAIModel", "ColPaliRetriever"]


def __getattr__(name):
    if name == "OpenAIModel":
        from .openai_model import OpenAIModel

        return OpenAIModel
    if name == "ColPaliRetriever":
        from .colpali import ColPaliRetriever

        return ColPaliRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
