import re
from ..constants import Patterns


def format_markdown(text: str) -> str:
    """Format markdown text by removing markdown and code blocks"""
    formatted_markdown = re.sub(Patterns.MATCH_MARKDOWN_BLOCKS, r"\1", text)
    formatted_markdown = re.sub(Patterns.MATCH_CODE_BLOCKS, r"\1", formatted_markdown)
    return formatted_markdown


def extract_html_content(text: str) -> str:
    """Extract and concatenate HTML content from code blocks"""
    matches = re.finditer(Patterns.MATCH_HTML_BLOCKS, text, re.MULTILINE | re.DOTALL)
    return "".join(match.group(1) for match in matches)


def get_final_html(html: str) -> str:
    """Get final HTML content by wrapping it in a basic HTML template"""
    final_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="p-2">
{html}
</body>
</html>
"""
    return final_html


# New function to extract JSON content
def extract_json_content(text: str) -> str:
    """Extract and concatenate JSON content from code blocks"""
    matches = re.finditer(Patterns.MATCH_JSON_BLOCKS, text, re.MULTILINE | re.DOTALL)
    return "".join(match.group(1) for match in matches)
