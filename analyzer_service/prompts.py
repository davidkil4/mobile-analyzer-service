import os
import logging
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# Get the directory where this script is located
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompt_file(filename: str) -> str:
    """Loads the content of a prompt file.

    Args:
        filename: The name of the file in the prompts directory.

    Returns:
        The content of the file as a string, or an empty string if an error occurs.
    """
    filepath = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filepath}")
        return ""
    except Exception as e:
        logger.error(f"Error reading prompt file {filepath}: {e}", exc_info=True)
        return ""


def load_correction_prompt() -> ChatPromptTemplate | None:
    """Loads the batch correction prompt and creates a ChatPromptTemplate.

    Returns:
        A ChatPromptTemplate object or None if loading fails.
    """
    prompt_content = load_prompt_file("batch_correction_prompt.txt")
    if not prompt_content:
        return None

    try:
        # Assuming the prompt takes 'batch_clauses_json' and 'context' as input variables
        # Adjust input_variables if the actual prompt template uses different names
        prompt_template = ChatPromptTemplate.from_template(
            prompt_content,
            # input_variables=["batch_clauses_json", "context"] # This might be needed depending on LC version
        )
        return prompt_template
    except Exception as e:
        logger.error(f"Error creating ChatPromptTemplate for correction prompt: {e}", exc_info=True)
        return None

# You can add similar loader functions for other prompts here
# e.g., load_segmentation_prompt, load_analysis_prompt, etc.
