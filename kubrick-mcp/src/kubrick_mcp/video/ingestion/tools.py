import base64
import subprocess
from io import BytesIO

import loguru
from moviepy import VideoFileClip
from PIL import Image

logger = loguru.logger.bind(name="VideoTools")


def extract_video_clip(video_path: str, start_time: float, end_time: float, output_path: str = None) -> VideoFileClip:
    # BUG: MoviePy crashes mid clip trimming. When it's got videos > N+5 minutes. Switching to ffmpeg for reliability.

    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")

    command = [
        "ffmpeg",
        "-ss",
        str(start_time),  # Seek (start time)
        "-to",
        str(end_time),  # End time
        "-i",
        video_path,  # Input file (after seek options)
        "-c:v",
        "libx264",  # Video codec (split into two arguments)
        "-preset",
        "medium",  # Add a preset for better balance of speed/quality
        "-crf",
        "23",  # Constant Rate Factor (quality setting, lower is better quality)
        "-c:a",
        "copy",  # Audio codec (copy stream without re-encoding)
        "-y",  # Overwrite output file without asking
        output_path,  # Output file (must be last)
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        logger.debug(f"FFmpeg output: {stdout.decode('utf-8', errors='ignore')}")
        return VideoFileClip(output_path)
    except subprocess.CalledProcessError as e:
        raise IOError(f"Failed to extract video clip: {str(e)}")


def encode_image(image: str | Image.Image) -> str:
    """Encode an image to base64 string.

    Args:
        image (Union[str, Image.Image]): Either a file path to an image or a PIL Image object

    Returns:
        str: Base64 encoded string representation of the image

    Raises:
        FileNotFoundError: If the image path does not exist
        IOError: If there are issues reading or processing the image
    """
    try:
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                image_str = image_file.read()
        else:
            if not image.format:
                image_format = "JPEG"
            else:
                image_format = image.format

            buffered = BytesIO()
            image.save(buffered, format=image_format)
            image_str = buffered.getvalue()

        return base64.b64encode(image_str).decode("utf-8")

    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Failed to process image: {str(e)}")


def decode_image(base64_string: str) -> Image.Image:
    """Decode a base64 string back into a PIL Image object.

    Args:
        base64_string (str): Base64 encoded string representation of an image

    Returns:
        Image.Image: PIL Image object

    Raises:
        ValueError: If the base64 string is invalid
        IOError: If there are issues processing the image data
    """
    try:
        image_bytes = base64.b64decode(base64_string)
        image_buffer = BytesIO(image_bytes)

        return Image.open(image_buffer)

    except (ValueError, IOError) as e:
        raise IOError(f"Failed to decode image: {str(e)}")
