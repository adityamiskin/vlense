__all__ = [
    "save_image",
    "encode_image_to_base64",
    "get_image_mime_type",
    "pdf_to_images",
    "format_markdown",
    "extract_html_content",
    "get_final_html",
    "process_batch_with_completion",
]


def __getattr__(name):
    if name in {"save_image", "encode_image_to_base64", "get_image_mime_type"}:
        from .image import encode_image_to_base64, get_image_mime_type, save_image

        return {
            "save_image": save_image,
            "encode_image_to_base64": encode_image_to_base64,
            "get_image_mime_type": get_image_mime_type,
        }[name]
    if name in {"pdf_to_images", "process_batch_with_completion"}:
        from .pdf import pdf_to_images, process_batch_with_completion

        return {
            "pdf_to_images": pdf_to_images,
            "process_batch_with_completion": process_batch_with_completion,
        }[name]
    if name in {"format_markdown", "extract_html_content", "get_final_html"}:
        from .text import extract_html_content, format_markdown, get_final_html

        return {
            "format_markdown": format_markdown,
            "extract_html_content": extract_html_content,
            "get_final_html": get_final_html,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
