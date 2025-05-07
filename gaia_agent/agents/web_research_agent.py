import os
from typing import List, Dict, Any, Optional

from smolagents import Tool
from smolagents.models import Model
from smolagents.monitoring import AgentLogger, LogLevel

from gaia_agent.agents.base import GAIAToolCallingAgent
from gaia_agent.config.settings import settings
from gaia_agent.tools.enhanced_search import (
    EnhancedGoogleSearchTool,
    EnhancedDuckDuckGoSearchTool,
    EnhancedWikipediaSearchTool
)
from gaia_agent.tools.text_web_browser import (
    SimpleTextBrowser,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool
)
from gaia_agent.utils.prompt_utils import load_prompt_templates
from gaia_agent.tools import TextInspectorTool


class WebResearchAgent(GAIAToolCallingAgent):
    """
    An agent specialized in performing robust web searches (Google, DDG, Wikipedia),
    visiting web pages, searching within pages, and accessing web archives.
    Includes rate limiting and retries for search tools. Called by the Manager Agent.
    """

    def __init__(
            self,
            model: Optional[Model] = None,
            tools: Optional[List[Tool]] = None,
            max_steps: Optional[int] = None,
            logger: Optional[AgentLogger] = None,
            browser_config: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        self.logger = logger or AgentLogger(level=LogLevel[settings.log_level.upper()])

        if model is None:
            self.logger.log(f"Initializing WebResearchAgent model: {settings.web_model_id}", level=LogLevel.DEBUG)
            from gaia_agent.utils.agent_setup import create_model_instance
            model = create_model_instance(settings.web_model_id, self.logger)

        if not browser_config:
            browser_config = self._setup_browser_config()

        browser = SimpleTextBrowser(**browser_config)

        if tools is None:
            tools = self._init_tools(browser, browser_config, model)

        # Setup agent prompt templates with web research specific additions
        agent_prompt_templates = self._setup_prompt_templates()

        # Initialize parent class
        super().__init__(
            name="web_researcher",
            description=self._get_agent_description(),
            model=model,
            tools=tools,
            prompt_templates=agent_prompt_templates,
            max_steps=max_steps or 20,
            logger=self.logger,
            verbosity_level=settings.default_verbosity_level,
            planning_interval=4,
            provide_run_summary=True,
            **kwargs
        )

        # Log initialization status
        self._log_initialization_status()

    def _setup_model(self, model: Optional[Model]) -> Model:
        """Initialize model if not provided."""
        if model is None:
            self.logger.log(f"Initializing WebResearchAgent model: {settings.web_model_id}", level=LogLevel.DEBUG)
            from utils.agent_setup import create_model_instance
            return create_model_instance(settings.web_model_id, self.logger)
        return model

    @staticmethod
    def _setup_browser_config() -> Dict[str, Any]:
        """Setup browser configuration with defaults if not provided."""
        browser_config = {
            "viewport_size": settings.browser_viewport_size,
            "downloads_folder": settings.browser_downloads_folder,
            "request_kwargs": {
                "headers": {"User-Agent": settings.browser_user_agent},
                "timeout": settings.browser_timeout,
            },
            # API keys needed for EnhancedGoogleSearchTool
            "serpapi_key": settings.serpapi_api_key or os.getenv("SERPAPI_API_KEY"),
            "serper_api_key": settings.serper_api_key or os.getenv("SERPER_API_KEY"),
        }
        os.makedirs(browser_config["downloads_folder"], exist_ok=True)
        return browser_config

    def _init_tools(self, browser: SimpleTextBrowser, browser_config: Dict[str, Any], model: Model) -> List[Tool]:
        """Initialize all tools needed for web research."""
        # Initialize search tools with fallback mechanisms
        search_tools = self._initialize_search_tools(browser_config)

        # Initialize browser navigation tools
        browser_tools = [
            VisitTool(browser=browser),
            PageUpTool(browser=browser),
            PageDownTool(browser=browser),
            FinderTool(browser=browser),
            FindNextTool(browser=browser),
            ArchiveSearchTool(browser=browser),
            EnhancedWikipediaSearchTool(),
            TextInspectorTool(model=model, text_limit=settings.doc_text_limit)
        ]

        self.logger.log("Initialized enhanced web tools.", level=LogLevel.INFO)
        return search_tools + browser_tools

    def _initialize_search_tools(self, browser_config: Dict[str, Any]) -> List[Tool]:
        """Initialize search tools with fallback mechanism and rate limiting."""
        search_tools = []
        has_google_key = False

        # Try Google Search (Serper first, then SerpApi)
        if browser_config.get("serper_api_key"):
            search_tools.append(EnhancedGoogleSearchTool(provider="serper"))
            self.logger.log("Initialized Enhanced Google Search Tool (Serper).", level=LogLevel.INFO)
            has_google_key = True
        elif browser_config.get("serpapi_key"):
            search_tools.append(EnhancedGoogleSearchTool(provider="serpapi"))
            self.logger.log("Initialized Enhanced Google Search Tool (SerpAPI).", level=LogLevel.INFO)
            has_google_key = True
        else:
            self.logger.log("No Serper or SerpApi key found for Google Search.", level=LogLevel.ERROR)

        # Always add DuckDuckGo
        search_tools.append(EnhancedDuckDuckGoSearchTool())
        self.logger.log("Initialized Enhanced DuckDuckGo Search Tool.", level=LogLevel.INFO)

        if not has_google_key:
            self.logger.log("Google Search tool unavailable, DuckDuckGo will be the primary search.",
                            level=LogLevel.ERROR)

        return search_tools

    @staticmethod
    def _setup_prompt_templates() -> Dict[str, Any]:
        """Set up agent prompt templates with web research specific additions."""
        agent_prompt_templates = load_prompt_templates(prompts_dir=settings.prompts_dir,
                                                       filename="gaia_toolcalling_agent.yaml")
        agent_prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
           If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
           Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
        return agent_prompt_templates

    @staticmethod
    def _get_agent_description() -> str:
        """Get the standard description for web research agent."""
        return """A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """

    def _log_initialization_status(self) -> None:
        """Log the agent's initialization status."""
        if self.tools:
            self.logger.log(f"WebResearchAgent initialized with tools: {list(dict(self.tools).keys())}.",
                            level=LogLevel.INFO)
        else:
            self.logger.log("WebResearchAgent initialized. WARNING: Some tools might be missing.", level=LogLevel.ERROR)