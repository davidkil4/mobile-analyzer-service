import re
import logging

logger = logging.getLogger(__name__)

def contains_korean(text: str | None) -> bool:
    """Checks if the given string contains any Korean Hangul characters.

    Args:
        text: The string to check.

    Returns:
        True if the string contains Korean characters, False otherwise.
    """
    if not text:
        return False
    # Check for Hangul Syllables Unicode range: AC00–D7AF
    # Includes Hangul Compatibility Jamo (U+3130–U+318F) and Syllables (U+AC00-U+D7A3)
    # Corrected pattern uses single backslash for unicode escape in raw string
    korean_pattern = re.compile(r"[\u3130-\u318F\uAC00-\uD7A3]")
    try:
        if korean_pattern.search(text):
            return True
    except Exception as e:
        # Log error if regex fails unexpectedly on the text
        logger.error(f"Error checking for Korean characters in text: '{text[:50]}...': {e}", exc_info=True)
    return False
