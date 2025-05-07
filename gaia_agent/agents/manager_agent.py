from typing import List

from smolagents import MultiStepAgent, Tool
from smolagents.models import Model
from smolagents.monitoring import AgentLogger, LogLevel

from gaia_agent.agents.base import GAIACodeAgent
from gaia_agent.agents.web_research_agent import WebResearchAgent
from gaia_agent.config.settings import settings



class ManagerAgent(GAIACodeAgent):
    """
    The orchestrating agent that receives the main task, potentially breaks it down,
    delegates sub-tasks to specialized agents (web researcher, document analyzer, calculator),
    synthesizes the results, and provides the final formatted answer.
    It also handles direct interaction with multi-modal tools (image, audio).
    """
    def __init__(
        self,
        model: Model | None = None,
        tools: List[Tool] | None = None, # Tools passed from setup
        managed_agents: List[MultiStepAgent] | None = None,
        max_steps: int | None = None,
        logger: AgentLogger | None = None,
        planning_interval: int = 4,
        **kwargs
    ):
        agent_logger = logger or AgentLogger(level=LogLevel[settings.log_level.upper()])
        if model is None:
            from gaia_agent.utils.agent_setup import create_model_instance
            agent_logger.log(f"Initializing ManagerAgent model: {settings.manager_model_id}", level=LogLevel.DEBUG)
            model = create_model_instance(settings.manager_model_id, agent_logger)


        if managed_agents is None:
            agent_logger.log("ManagerAgent: Initializing default managed agents...", level=LogLevel.INFO)
            web_agent = WebResearchAgent(logger=agent_logger)
            managed_agents = [web_agent]
            agent_logger.log("ManagerAgent: Default managed agents initialized.", level=LogLevel.INFO)
        else:
             agent_logger.log(f"ManagerAgent: Using pre-initialized managed agents: {[a.name for a in managed_agents]}", level=LogLevel.INFO)

        super().__init__(
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            max_steps=max_steps or settings.default_agent_max_steps,
            additional_authorized_imports=["*"],
            verbosity_level=settings.default_verbosity_level,
            logger=agent_logger,
            planning_interval=planning_interval,
            **kwargs
        )
        agent_logger.log(f"ManagerAgent initialized with tools: {list(dict(self.tools).keys())}.", level=LogLevel.INFO)
        agent_logger.log(f"ManagerAgent initialized with managed agents: {list(dict(self.managed_agents).keys())}.", level=LogLevel.INFO)

