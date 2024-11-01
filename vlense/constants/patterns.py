class Patterns:
    """Regex patterns for markdown and code blocks"""

    MATCH_MARKDOWN_BLOCKS = r"```[a-z]*\n([\s\S]*?)\n```"

    MATCH_CODE_BLOCKS = r"```(?:[a-z]*)\n([\s\S]*?)\n```"

    # Updated pattern without ^ and $ to match multiple HTML blocks
    MATCH_HTML_BLOCKS = r"```html\n([\s\S]*?)\n```"

    MATCH_JSON_BLOCKS = r"```json\n([\s\S]*?)\n```"
