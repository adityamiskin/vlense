class Messages:
    """User-facing messages"""

    MISSING_ENVIRONMENT_VARIABLES = """
    Required OpenAI-compatible credentials are missing. Please set OPENAI_API_KEY.
    You can also optionally set OPENAI_BASE_URL for a compatible gateway.
    """

    NON_VISION_MODEL = """
    The provided model is not a vision model. Please provide a vision model.
    """

    MODEL_ACCESS_ERROR = """
    Your provided model can't be accessed. Please make sure you have access to the model and that OPENAI_API_KEY or OPENAI_BASE_URL are configured correctly.
    """

    CUSTOM_SYSTEM_PROMPT_WARNING = """
    Custom system prompt was provided which overrides the default system prompt. We assume that you know what you are doing.  
    """

    MAINTAIN_FORMAT_SELECTED_PAGES_WARNING = """
    The maintain_format flag is set to True in conjunction with select_pages input given. This may result in unexpected behavior.
    """

    PAGE_NUMBER_OUT_OF_BOUND_ERROR = """
    The page number(s) provided is out of bound. Please provide a valid page number(s).
    """

    NON_200_RESPONSE = """
    Model API returned status code {status_code}: {data}
    """

    COMPLETION_ERROR = """
    Error in Completion Response. Error: {0}
    Please check the status of your model provider API status.
    """

    PDF_CONVERSION_FAILED = """
    Error during PDF conversion: {0}
    Please check the PDF file and try again. For more information: https://github.com/Belval/pdf2image
    """

    FILE_UNREACHAGBLE = """
    File not found or unreachable. Status Code: {0}
    """

    FILE_PATH_MISSING = """
    File path is invalid or missing.
    """

    FAILED_TO_SAVE_FILE = """Failed to save file to local drive"""

    FAILED_TO_PROCESS_IMAGE = """Failed to process image"""
