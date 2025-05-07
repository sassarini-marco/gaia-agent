from functools import lru_cache
from pathlib import Path

import yaml

from smolagents.agents import PromptTemplates, EMPTY_PROMPT_TEMPLATES
from smolagents.monitoring import AgentLogger, LogLevel

util_logger = AgentLogger(level=LogLevel.INFO)

@lru_cache(maxsize=None)
def load_prompt_templates(
    prompts_dir: str | Path, # Use str/Path as cache key
    filename: str
    ) -> PromptTemplates:
    """
    Loads prompt templates from a YAML file within the specified directory.
    Uses lru_cache to load from disk only once per unique directory/filename combination.

    Args:
        prompts_dir (str | Path): The directory containing the prompt file.
        filename (str): The name of the YAML file (default: gaia_level1_prompts.yaml).

    Returns:
        PromptTemplates: A dictionary containing the loaded prompts, or
                         EMPTY_PROMPT_TEMPLATES if loading fails.
    """
    logger = util_logger
    prompts_file_path = Path(prompts_dir) / filename

    try:
        logger.log(f"Attempting to load prompts from: {prompts_file_path} (cache miss or first call)", level=LogLevel.DEBUG)
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            loaded_prompts = yaml.safe_load(f)

        logger.log(f"Successfully loaded and cached prompts from: {prompts_file_path}", level=LogLevel.INFO)

        return loaded_prompts

    except FileNotFoundError:
        logger.log(
            f"Error: Prompts file not found at configured path: {prompts_file_path}. Using default empty prompts.",
            level=LogLevel.ERROR
        )
        return EMPTY_PROMPT_TEMPLATES

    except yaml.YAMLError as e:
        logger.log(
            f"Error parsing prompts file {prompts_file_path}: {e}. Using default empty prompts.",
            level=LogLevel.ERROR
        )
        return EMPTY_PROMPT_TEMPLATES

    except Exception as e:
         logger.log(
            f"An unexpected error occurred loading prompts from {prompts_file_path}: {e}. Using default empty prompts.",
            level=LogLevel.ERROR
         )
         return EMPTY_PROMPT_TEMPLATES