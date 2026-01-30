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


def get_extension_from_mime_type(mime_type: str) -> str:
    """Get file extension from MIME type.

    Args:
        mime_type: MIME type string (e.g., "image/png", "image/jpeg").

    Returns:
        File extension including the dot (e.g., ".png", ".jpg").
    """
    mime_to_ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mime_to_ext.get(mime_type.lower(), ".png")


async def save_image_to_disk(
    image_data: bytes, filename: str, output_dir: str | None = None, mime_type: str = "image/png"
) -> str:
    """Save raw image bytes to disk with the correct extension based on mime type.

    Args:
        image_data: Raw image bytes.
        filename: Name for the output file (without extension).
        output_dir: Optional directory to save the image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.
        mime_type: MIME type of the image (e.g., "image/png", "image/jpeg").
                   Used to determine the file extension.

    Returns:
        Absolute path to the saved image file.

    Raises:
        Exception: If the image cannot be opened or saved.
    """
    try:
        # Use provided output_dir, fallback to DEFAULT_OUTPUT_IMAGE_PATH
        target_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_IMAGE_PATH

        # Ensure directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension from mime type
        extension = get_extension_from_mime_type(mime_type)

        # Save file
        image = PIL.Image.open(io.BytesIO(image_data))
        image_path = target_dir / f"{filename}{extension}"
        image.save(image_path)
        logger.info(f"Image saved to {image_path} (mime_type: {mime_type})")
        return str(image_path)
    except Exception as e:
        logger.error(f"Error saving image: {e!s}")
        raise
