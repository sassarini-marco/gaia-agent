from typing import TYPE_CHECKING

from smolagents.default_tools import SpeechToTextTool
from smolagents.models import LiteLLMModel, Model, InferenceClientModel
from smolagents.monitoring import AgentLogger, LogLevel

from gaia_agent.agents import WebResearchAgent
from gaia_agent.config.settings import settings
from gaia_agent.tools import TextInspectorTool

try:
    from gaia_agent.tools.visual_qa import visualizer
except ImportError:
    print("WARNING: Visualizer tool not found. Image analysis will not be available.")
    visualizer = None

if TYPE_CHECKING:
    from gaia_agent.agents import ManagerAgent


def _get_api_key_for_model(model_id: str) -> str | None:
    """Helper to select the correct API key based on model ID prefix."""
    assert settings is not None, "Settings not loaded"
    model_id_lower = model_id.lower()

    if model_id_lower.startswith("gemini") or model_id_lower.startswith("google"):
        secret = settings.google_api_key
        if not secret:
            raise ValueError("GOOGLE_API_KEY required for Gemini/Google models but not set.")
        return secret.get_secret_value()

    elif model_id_lower.startswith("claude") or model_id_lower.startswith("anthropic"):
        secret = settings.anthropic_api_key
        if not secret:
            raise ValueError("ANTHROPIC_API_KEY required for Anthropic models but not set.")
        return secret.get_secret_value()

    elif model_id_lower.startswith("gpt-") or model_id_lower.startswith("openai"):
        secret = settings.openai_api_key
        if not secret:
            raise ValueError("OPENAI_API_KEY required for OpenAI models but not set.")
        return secret.get_secret_value()

    return None

def create_model_instance(model_id: str, logger: AgentLogger, model_type: str = "LiteLLMModel") -> Model:
    """Creates a model instance based on ID and configured type."""
    logger.log(f"Creating {model_type} instance for ID: {model_id}", level=LogLevel.DEBUG)
    api_key = _get_api_key_for_model(model_id)

    api_base = settings.openai_api_base if settings.openai_api_base and ("gpt-" in model_id or "openai" in model_id) else None

    try:
        if model_type == "LiteLLMModel":
            model_instance = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base,
                temperature=settings.default_temperature,
                max_tokens=settings.default_max_tokens
            )

        elif model_type == "InferenceClientModel":
             hf_token = settings.huggingfacehub_api_token.get_secret_value() if settings.huggingfacehub_api_token else None
             if not hf_token:
                 logger.log("HUGGINGFACEHUB_API_TOKEN not set, InferenceClientModel might fail for gated/private models.", level=LogLevel.ERROR)
             model_instance = InferenceClientModel(
                 model_id=model_id,
                 token=hf_token,
                 temperature=settings.default_temperature,
             )
        else:
            raise ValueError(f"Unsupported model_type in create_model_instance: {model_type}")

        logger.log(f"Successfully created {model_type} for {model_id}.", level=LogLevel.DEBUG)
        return model_instance
    except Exception as e:
        logger.log(f"Failed to create model instance for {model_id} (Type: {model_type}): {e}", level=LogLevel.ERROR)
        raise RuntimeError(f"Could not instantiate model: {model_id}") from e


def _initialize_logger():
    """Initialize and configure the system logger."""
    if not settings:
        raise RuntimeError("Application settings failed to load.")

    log_level_enum = LogLevel[settings.log_level.upper()]
    logger = AgentLogger(level=log_level_enum)
    logger.log("Setting up GAIA Agent System", level=LogLevel.INFO)
    return logger


def _initialize_manager_tools(manager_model, logger):
    """Initialize tools for the manager agent."""
    logger.log("Initializing tools...", level=LogLevel.DEBUG)
    manager_tools = []

    # file to text
    if TextInspectorTool:
        manager_tools.append(TextInspectorTool(model=manager_model, text_limit=settings.doc_text_limit))
        logger.log("Added TextInspectorTool to manager agent.", level=LogLevel.INFO)
    else:
        logger.log("TextInspectorTool not available.", level=LogLevel.ERROR)

    # image to text
    if visualizer:
        manager_tools.append(visualizer)
        logger.log("Added Visualizer Tool.", level=LogLevel.INFO)
    else:
        logger.log("Visualizer tool not available.", level=LogLevel.ERROR)

    # speech-to-text
    try:
        manager_tools.append(SpeechToTextTool())
        logger.log("Added SpeechToText Tool.", level=LogLevel.INFO)
    except Exception as e:
        logger.log(f"Failed to initialize SpeechToTextTool: {e}", level=LogLevel.ERROR)

    return manager_tools


def _initialize_managed_agents(logger):
    """Initialize all managed agents needed by the system."""
    logger.log("Initializing manager managed agents...", level=LogLevel.DEBUG)

    web_research_agent = WebResearchAgent(logger=logger)
    managed_agents = [web_research_agent]

    if not managed_agents:
        logger.log("CRITICAL: No specialized agents could be initialized.", level=LogLevel.ERROR)
        raise RuntimeError("Failed to initialize any specialized agents.")

    return managed_agents



def setup_agent_system() -> "ManagerAgent": # <-- Use string literal for return type
    """
    Loads configuration, prompts, tools and instantiates the complete agent system.
    Returns the top-level ManagerAgent instance.
    """
    from gaia_agent.agents import ManagerAgent

    logger = _initialize_logger()
    logger.log("Initializing tools...", level=LogLevel.DEBUG)

    manager_model = create_model_instance(settings.manager_model_id, logger)
    manager_tools = _initialize_manager_tools(manager_model, logger)
    managed_agents = _initialize_managed_agents(logger)

    try:
        manager_agent = ManagerAgent(
            model=manager_model,
            tools=manager_tools,
            managed_agents=managed_agents,
            logger=logger
        )
    except Exception as e:
        logger.log(f"Fatal error during manager agent initialization: {e}", level=LogLevel.ERROR)
        raise RuntimeError("Failed to initialize ManagerAgent") from e

    logger.log("Agent system setup complete.", level=LogLevel.INFO)
    return manager_agent