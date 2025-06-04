import opik
from loguru import logger

from mcp_server.config import get_settings

settings = get_settings()
opik.configure(api_key=settings.OPIK_API_KEY, workspace=settings.OPIK_WORKSPACE)

client = opik.Opik(
    project_name=settings.OPIK_PROJECT,
)
logger = logger.bind(name="Prompts")


def system_prompt() -> str:
    try:
        prompt = client.get_prompt(name="prompt-system")
        logger.info(f"System prompt loaded. \n {prompt.commit=} \n {prompt.prompt=}")
        return prompt.prompt
    except Exception:
        logger.warning("Nada, not working. Check opik creds playa")
        prompt = "You are a helpful assistant. You need to help the user solve some sport related problems. Your input is {{input}}"
    return prompt


def struct_output_prompt() -> str:
    try:
        prompt = client.get_prompt(name="prompt-struct-output")
        return prompt.prompt
    except Exception:
        logger.warning("Nada, not working. Check opik creds playa")
        prompt = "You are a helpful assistant. You need to help the user solve some sport related problems. Your output is {{output}}"
    return prompt


if __name__ == "__main__":
    # For testing purposes
    print("System Prompt:", system_prompt())
    print("Structured Output Prompt:", struct_output_prompt())
