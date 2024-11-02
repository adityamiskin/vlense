class Prompts:
    """Class for storing prompts for the Zerox system."""

    DEFAULT_SYSTEM_PROMPT_MARKDOWN = """
    Convert the following PDF page to markdown.
    Return only the markdown with no explanation text.
    Do not exclude any content from the page. Give the markdown in ```markdown``` format.
    """

    DEFAULT_SYSTEM_PROMPT_HTML = """
    Analyze this image and create an HTML representation using Tailwind CSS classes that recreates the layout and content exactly as shown.
    Focus on:
    1. Exact positioning and spacing
    2. Correct font sizes and styles
    3. Colors and backgrounds
    4. Any visible elements like borders, shadows, etc.
    5. Write only the HTML code for the content inside the body tag as everything else is provided.
    
    Write only in the html_content variable as ```html
    {html_content}
    ```
    
    Provide only the HTML code without any explanations. Use semantic HTML elements where appropriate.
    """

    # New default system prompt for JSON schema mode
    DEFAULT_SYSTEM_PROMPT_JSON = """
    Extract the event information and structure it according to the provided JSON schema.
    Return only the JSON with no explanation text.
    Do not exclude any content from the page.
    """
