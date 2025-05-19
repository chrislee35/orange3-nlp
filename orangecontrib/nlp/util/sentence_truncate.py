import re

def truncate_at_sentence(text: str, max_chars: int = 3700) -> str:
    """
    Truncate `text` at the last complete sentence before `max_chars`.

    Args:
        text (str): The input text.
        max_chars (int): The maximum allowed number of characters.

    Returns:
        str: The truncated text ending at the last full sentence.
    """
    if len(text) <= max_chars:
        return text

    # Cut to the max limit
    snippet = text[:max_chars]

    # Find the last sentence-ending punctuation before the cut
    match = re.search(r'(?s)(.*?[\.\!\?])[^\.!?]*$', snippet)
    if match:
        return match.group(1).strip()
    else:
        # If no sentence boundary found, fall back to word boundary
        return snippet.rsplit(' ', 1)[0].strip()
