from groq import Groq
from loguru import logger
from models import UserContent
from PIL import Image
from tools import encode_image

from mcp_server.config import get_settings

settings = get_settings()


class VisualCaptioningModel:
    """A model for generating captions from images using Groq's vision-language models.

    This class provides functionality to caption images using Groq's VLM models through their API.

    Args:
        model_name (str): The name of the Groq model to use. Defaults to settings.GROQ_VLM_MODEL.

    Attributes:
        model_name (str): The name of the Groq model being used.
        client (Groq): The Groq API client instance.
    """

    def __init__(self, model_name: str = settings.GROQ_VLM_MODEL):
        self.model_name = model_name
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def caption(self, image: Image.Image | str, prompt: str, verbose: bool = False) -> str:
        """Generate a caption for the given image using the specified prompt.

        Args:
            image (Union[Image, str]): The input image, either as a PIL Image object or a path string.
            prompt (str): The prompt text to guide the caption generation.

        Returns:
            str: The generated caption text.

        Note:
            If image is provided as a string, it will be loaded and converted to RGB format.
        """
        base64_image = encode_image(image)
        message = UserContent.from_pair(
            image=base64_image,
            prompt=prompt,
        )
        chat_completion = self.client.chat.completions.create(
            messages=[message.model_dump(by_alias=True)],
            model=self.model_name,
        )

        if verbose:
            logger.info("Image successfully captioned")

        return chat_completion.choices[0].message.content

    def __repr__(self):
        return f"VisualCaptioningModel(model_name={self.model_name})"
