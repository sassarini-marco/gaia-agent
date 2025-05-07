import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent

ENV_FILE_PATHS = (
    PROJECT_ROOT / ".env",  # 1. Check project root directory
    PROJECT_ROOT.parent / ".env",  # 2. Check parent directory
)


class AppSettings(BaseSettings):
    """
    Application settings. Loads from environment variables and .env file.
    Environment variables override values from .env file.
    """
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATHS,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- API Keys ---
    google_api_key: Optional[SecretStr] = Field(None, description="API Key for Google Gemini models.")
    google_cloud_project: Optional[str] = Field(None,
                                                description="Google Cloud Project ID required for Vertex AI/Gemini.")
    # todo other providers added but not tested. Stick to gemini for now.
    anthropic_api_key: Optional[SecretStr] = Field(None, description="API Key for Anthropic Claude models.")
    huggingface_token: Optional[SecretStr] = Field(None,
                                                   description="Hugging Face Hub token (for API or private models).")
    openai_api_key: Optional[SecretStr] = Field(None,
                                                description="API Key for OpenAI or compatible APIs (e.g., Fireworks).")
    serpapi_api_key: Optional[SecretStr] = Field(None, description="API Key for SERPAPI Google Search API.")
    serper_api_key: Optional[SecretStr] = Field(None, description="API Key for Serper.dev Google Search API.")

    # --- Browser Configuration ---
    browser_user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        description="User agent string for browser requests."
    )
    browser_viewport_size: int = Field(
        5120,
        description="Viewport size for browser (width in pixels)."
    )
    browser_downloads_folder: str = Field(
        "downloads_folder",
        description="Folder path for browser downloads."
    )
    browser_timeout: int = Field(
        300,
        description="Timeout for browser requests in seconds."
    )

    # --- Model Configuration ---
    manager_model_id: str = Field("gemini-2.0-flash", description="Model ID for the Manager Agent (orchestration).")
    web_model_id: str = Field("gemini-2.0-flash", description="Model ID for the Web Research Agent.")
    text_inspector_model_id: str = Field("gemini-2.0-flash", description="Model ID for the text ispector")
    openai_api_base: Optional[str] = Field(None, description="Base URL for OpenAI compatible API, if not default.")

    # --- Agent Defaults ---
    default_agent_max_steps: int = Field(10, description="Default maximum steps for agents.")
    default_temperature: float = Field(0.0,
                                       description="Default sampling temperature for LLM generation (0.0 for deterministic).")
    default_verbosity_level: int = Field(2,
                                       description="Default verbosity level")
    default_max_tokens: int = Field(2000,
                                       description="Default max token")
    doc_text_limit: int = Field(100000, description="Text limit to inspect")

    # --- File Paths ---
    project_root: str = str(PROJECT_ROOT.resolve())
    prompts_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "prompts",
        description="Directory containing agent prompt YAML files."
    )
    gaia_data_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "gaia",
        description="Directory to store/load GAIA dataset files."
    )

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level for the application."
    )

    # --- Benchmark API (for submission script) ---
    benchmark_api_url: str = Field(
        "https://default-benchmark-api.hf.space",
        description="URL for the benchmark submission API server.",
    )

    # Add a validator to ensure project ID is set if a gemini model is used
    @field_validator('manager_model_id', 'web_model_id', 'text_inspector_model_id')
    @classmethod
    def check_gemini_project(cls, v, info):
        if "gemini" in v and not info.data.get('google_cloud_project'):
            raise ValueError(
                f"Model '{v}' requires 'google_cloud_project' to be set in settings or GOOGLE_CLOUD_PROJECT env var.")
        return v


# Instantiate settings
try:
    settings = AppSettings()

    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_CONTENT'):
        from gaia_agent.utils.gcp_credentials import setup_google_credentials

        setup_google_credentials()

    # Find which .env file was actually loaded (if any)
    loaded_env_file = None
    for p in ENV_FILE_PATHS:
        if p.is_file():
            loaded_env_file = p
            break

    if loaded_env_file:
        print(f"Loaded settings from: {loaded_env_file}")
    else:
        print("No .env file found in search paths.")

    if "gemini" in settings.manager_model_id and not settings.google_cloud_project:
        print("Warning: Manager Agent uses Gemini, but GOOGLE_CLOUD_PROJECT seems missing.")

    if not settings.prompts_dir.is_dir():
        print(f"Warning: Default prompts directory does not exist: {settings.prompts_dir}")

    print(
        f"Settings loaded: Manager model='{settings.manager_model_id}', Prompts Dir='{os.path.abspath(settings.prompts_dir)}'")

    loaded_keys = [k for k, v in settings.model_dump().items() if ("api_key" in k or "token" in k) and v is not None]
    if not loaded_keys:
        print("Warning: No API keys/tokens seem to be loaded from environment or .env file.")
    else:
        print(f"Found API keys/tokens for: {', '.join(loaded_keys)}")

except ValidationError as e:
    print(f"Error loading settings: {e}")
    settings = None