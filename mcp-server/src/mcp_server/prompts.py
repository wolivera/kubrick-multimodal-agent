import opik
from loguru import logger

client = opik.Opik()

logger = logger.bind(name="Prompts")


ROUTING_SYSTEM_PROMPT = """
You are a routing assistant part of a video processing application. 
You need to determine if the user query requires the use of a tool. 
The tools available are:

- get_video_clip_from_user_query: This tool is used to get a clip from the video based on the user query.
- ask_question_about_video: This tool is used to ask a question about the video.

You output should be a boolean value indicating it tool use is required or not.
"""

TOOL_USE_SYSTEM_PROMPT = """
Your name is Kubrick.
You are a tool use assistant part of a video processing application.
You need to determine which tool to use based on the user query (if any).

The tools available are

- get_video_clip_from_user_query: This tool is used to get a clip from the video based on the user query.
- ask_question_about_video: This tool is used to get some information about the video.
"""

GENERAL_SYSTEM_PROMPT = """
Your name is Kubrick. 
You are a general assistant part of a video processing application.
You need to help the user with their query.
"""


def routing_system_prompt() -> str:
    try:
        prompt = client.get_prompt(name="routing-system-prompt")
        logger.info(f"System prompt loaded. \n {prompt.commit=} \n {prompt.prompt=}")
        return prompt.prompt
    except Exception:
        logger.warning("Nada, not working. Check opik creds playa")
        prompt = ROUTING_SYSTEM_PROMPT
    return prompt


def tool_use_system_prompt() -> str:
    try:
        prompt = client.get_prompt(name="tool-use-system-prompt")
        return prompt.prompt
    except Exception:
        logger.warning("Nada, not working. Check opik creds playa")
        prompt = TOOL_USE_SYSTEM_PROMPT
    return prompt


def general_system_prompt() -> str:
    try:
        prompt = client.get_prompt(name="general-system-prompt")
        return prompt.prompt
    except Exception:
        logger.warning("Nada, not working. Check opik creds playa")
        prompt = GENERAL_SYSTEM_PROMPT
    return prompt
