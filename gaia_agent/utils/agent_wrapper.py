import os
import shutil
import textwrap
from typing import Any, Dict

from smolagents import AgentError

from gaia_agent.config.settings import settings
from gaia_agent.tools import TextInspectorTool
from gaia_agent.tools.visual_qa import visualizer
from gaia_agent.utils.agent_setup import setup_agent_system, create_model_instance


class GAIABenchmarkAgentWrapper:
    """Wraps an agent instance to automatically clean its final response."""
    def __init__(self):

        self._agent = setup_agent_system()
        self.logger = getattr(self._agent, 'logger', None)

    def __call__(self, sample: Dict[str, Any], **kwargs) -> str:
        """
        Executes the wrapped agent's run method and cleans the output.
        """
        if hasattr(self._agent, 'run') and callable(self._agent.run):
            task = self._prepare_task(sample)
            raw_output = self._agent.run(task, **kwargs)
        else:
            raise AttributeError("Wrapped agent object does not have a callable 'run' method.")

        cleaned_output = self._clean_agent_answer(raw_output)

        if self.logger:
             self.logger.log(f"AgentWrapper raw output: '{raw_output}', Cleaned output: '{cleaned_output}'", level=1) # Use appropriate level

        return cleaned_output


    @staticmethod
    def _clean_agent_answer(raw_answer: str | Any) -> str:
        """Removes known prefixes from the agent's final answer."""
        answer_str = str(raw_answer)  # Ensure it's a string
        prefixes_to_remove = [
            "Out - Final answer: ",
            "Final Answer: ",
            "FINAL ANSWER: ",
            "### Core Answer Value: ",
            "The answer is ",
            "Answer: ",
            "The result is ",
            "To answer this question: ",
            "Based on the information provided, ",
            "According to the information: "
        ]
        for prefix in prefixes_to_remove:
            if answer_str.startswith(prefix):
                return answer_str[len(prefix):].strip()

        # If no known prefix is found, return the stripped original string
        return answer_str.strip()

    def interrupt(self):
        """Calls the interrupt method of the wrapped agent."""
        if hasattr(self._agent, 'interrupt') and callable(self._agent.interrupt):
            print("Interrupt signal sent to agent.")
            self._agent.interrupt()
        else:
            print("Warning: Wrapped agent does not have an interrupt method.")


    def reset(self, *args, **kwargs):
        if hasattr(self._agent, 'reset') and callable(self._agent.reset):
            return self._agent.reset(*args, **kwargs)
        return None

    def _prepare_task(self,
            sample: Dict[str, Any]
    ) -> str:
        """
        Adapted from https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py
        """
        question = sample.get("question", sample.get('Question'))
        file_name = sample.get("file_name")
        assert question is not None, "missing question key from given sample"

        # todo not very efficient recreate the model every time. Keep to make sure to start from fresh context each time.
        text_inspector_model = create_model_instance(settings.text_inspector_model_id, self.logger)
        document_inspection_tool = TextInspectorTool(text_inspector_model, settings.doc_text_limit)
        augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
    Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
    Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
    Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

    """ + question

        if file_name:
            if ".zip" in file_name:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += get_zip_description(
                    file_name, question, visualizer, document_inspection_tool
                )
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += get_single_file_description(
                    file_name, question, visualizer, document_inspection_tool
                )
            augmented_question += prompt_use_files

        return augmented_question



# From here adapting method from https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py
def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


def get_image_description(file_name: str, question: str, visual_inspection_tool) -> str:
    prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the image."""
    return visual_inspection_tool(image_path=file_name, question=prompt)


def get_document_description(file_path: str, question: str, document_inspection_tool) -> str:
    prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the document."""
    return document_inspection_tool.forward_initial_exam_mode(file_path=file_path, question=prompt)


def get_single_file_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    file_extension = file_path.split(".")[-1]
    if file_extension in ["png", "jpg", "jpeg"]:
        file_description = f" - Attached image: {file_path}"
        file_description += (
            f"\n     -> Image description: {get_image_description(file_path, question, visual_inspection_tool)}"
        )
        return file_description
    elif file_extension in ["pdf", "xls", "xlsx", "docx", "doc", "xml"]:
        image_path = file_path.split(".")[0] + ".png"
        if os.path.exists(image_path):
            description = get_image_description(image_path, question, visual_inspection_tool)
            file_path = image_path
        else:
            description = get_document_description(file_path, question, document_inspection_tool)
        file_description = f" - Attached document: {file_path}"
        file_description += f"\n     -> File description: {description}"
        return file_description
    elif file_extension in ["mp3", "m4a", "wav"]:
        return f" - Attached audio: {file_path}"
    else:
        return f" - Attached file: {file_path}"


def get_zip_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    folder_path = file_path.replace(".zip", "")
    os.makedirs(folder_path, exist_ok=True)
    shutil.unpack_archive(file_path, folder_path)

    prompt_use_files = ""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            prompt_use_files += "\n" + textwrap.indent(
                get_single_file_description(file_path, question, visual_inspection_tool, document_inspection_tool),
                prefix="    ",
            )
    return prompt_use_files