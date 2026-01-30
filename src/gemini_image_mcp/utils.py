import base64
import io
import logging
import os
from pathlib import Path

import PIL.Image

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_IMAGE_PATH = Path(os.getenv("DEFAULT_OUTPUT_IMAGE_PATH") or Path.cwd())


def is_valid_base64_image(base64_string: str) -> bool:
    """Check if a string is a valid base64-encoded image.

    Args:
        base64_string: Base64-encoded string to validate.

    Returns:
        True if the string decodes to a valid image, False otherwise.
    """
    try:
        image_data = base64.b64decode(base64_string)
        with PIL.Image.open(io.BytesIO(image_data)) as img:
            logger.debug(f"Validated base64 image, format: {img.format}, size: {img.size}")
            return True
    except Exception as e:
        logger.warning(f"Invalid base64 image: {e!s}")
        return False


async def save_image_to_disk(image_data: bytes, filename: str, output_dir: str | None = None) -> str:
    """Save raw image bytes to disk as a PNG file.

    Args:
        image_data: Raw image bytes.
        filename: Name for the output file (without extension).
        output_dir: Optional directory to save the image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        Exception: If the image cannot be opened or saved.
    """
    try:
        # Use provided output_dir, fallback to DEFAULT_OUTPUT_IMAGE_PATH
        target_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_IMAGE_PATH

        # Ensure directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        image = PIL.Image.open(io.BytesIO(image_data))
        image_path = target_dir / f"{filename}.png"
        image.save(image_path)
        logger.info(f"Image saved to {image_path}")
        return str(image_path)
    except Exception as e:
        logger.error(f"Error saving image: {e!s}")
        raise
