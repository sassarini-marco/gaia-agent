from smolagents import CodeAgent, ToolCallingAgent

from gaia_agent.config.settings import settings
from gaia_agent.utils.prompt_utils import load_prompt_templates


class GAIACodeAgent(CodeAgent):
    """
    A specialized implementation of the `CodeAgent` class tailored for the GAIA system.
    """

    def __init__(
        self,
        **kwargs
    ):

        if not kwargs.get("prompt_templates"):
            kwargs["prompt_templates"] = load_prompt_templates(prompts_dir=settings.prompts_dir,
                                                               filename="gaia_code_agent.yaml")
        super().__init__(
            **kwargs
        )


class GAIAToolCallingAgent(ToolCallingAgent):
    """
    A specialized implementation of the `ToolCallingAgent` class tailored for the GAIA system.
    """

    def __init__(
        self,
        **kwargs
    ):
        if not kwargs.get("prompt_templates"):
            kwargs["prompt_templates"] = load_prompt_templates(prompts_dir=settings.prompts_dir,
                                                               filename="gaia_toolcalling_agent.yaml")
        super().__init__(
            **kwargs
        )

