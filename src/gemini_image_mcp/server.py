import base64
import binascii
import logging
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import PIL.Image
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

from .prompts import (
    build_generation_prompt,
    build_transformation_prompt,
    build_translation_prompt,
)
from .utils import save_image_to_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("gemini-image-mcp")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
)


async def invoke_gemini_api(
    contents: list[Any],
    model: str | None = None,
    config: types.GenerateContentConfig | None = None,
    text_only: bool = False,
) -> str | bytes:
    """Call the Gemini API and return text or image data.

    Args:
        contents: Content to send (text and/or images).
        model: Gemini model name. Defaults to GEMINI_MODEL.
        config: Optional generation config.
        text_only: If True, return text response; otherwise return image bytes.

    Returns:
        Text string if text_only=True, otherwise raw image bytes.

    Raises:
        ValueError: If API key is missing or response lacks expected content.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    if model is None:
        model = GEMINI_MODEL

    http_options = types.HttpOptions(base_url=GEMINI_BASE_URL)
    client = genai.Client(api_key=api_key, http_options=http_options)

    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e!s}")
        raise

    logger.info(f"Response received from Gemini API using model {model}")

    if text_only:
        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            raise ValueError("No text content found in Gemini response")
        text = response.candidates[0].content.parts[0].text
        if text is None:
            raise ValueError("No text content found in Gemini response")
        return text.strip()

    if (
        response.candidates
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None and part.inline_data.data is not None:
                return part.inline_data.data

    raise ValueError("No image data found in Gemini response")


async def generate_filename_from_prompt(prompt: str) -> str:
    """Generate a concise filename from an image prompt using Gemini.

    Args:
        prompt: The image generation prompt.

    Returns:
        A short underscore-separated filename without extension.
    """
    try:
        filename_prompt = f"""
        Based on this image description: "{prompt}"

        Generate a short, descriptive file name suitable for the requested image.
        The filename should:
        - Be concise (maximum 5 words)
        - Use underscores between words
        - Not include any file extension
        - Only return the filename, nothing else
        """

        result = await invoke_gemini_api([filename_prompt], text_only=True)
        assert isinstance(result, str), "Expected string response"
        logger.info(f"Generated filename: {result}")
        return result

    except Exception as e:
        logger.error(f"Error generating filename with Gemini: {e!s}")
        truncated_text = prompt[:12].strip()
        return f"image_{truncated_text}_{str(uuid.uuid4())[:8]}"


async def translate_to_english(text: str) -> str:
    """Translate a prompt to English for better image generation.

    Args:
        text: Original prompt in any language.

    Returns:
        English translation, or original text if translation fails.
    """
    try:
        prompt = build_translation_prompt(text)
        result = await invoke_gemini_api([prompt], text_only=True)
        assert isinstance(result, str), "Expected string response"
        logger.info(f"Original prompt: {text}")
        logger.info(f"Translated prompt: {result}")
        return result

    except Exception as e:
        logger.error(f"Error translating prompt: {e!s}")
        return text


async def generate_and_save_image(
    contents: list[Any],
    prompt: str,
    model: str | None = None,
    output_dir: str | None = None,
) -> tuple[bytes, str]:
    """Generate an image with Gemini and save it to disk.

    Args:
        contents: Content list for Gemini (prompt and optional source image).
        prompt: Original prompt for filename generation.
        model: Optional Gemini model override.
        output_dir: Optional directory to save the image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Tuple of (image bytes, saved file path).
    """
    gemini_response = await invoke_gemini_api(
        contents,
        model=model,
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )
    assert isinstance(gemini_response, bytes), "Expected bytes response"

    filename = await generate_filename_from_prompt(prompt)
    saved_image_path = await save_image_to_disk(gemini_response, filename, output_dir)

    return gemini_response, saved_image_path


async def transform_and_save_image(
    source_image: PIL.Image.Image,
    optimized_edit_prompt: str,
    original_edit_prompt: str,
    output_dir: str | None = None,
) -> tuple[bytes, str]:
    """Transform an image with Gemini and save the result.

    Args:
        source_image: PIL Image to transform.
        optimized_edit_prompt: Translated/optimized transformation prompt.
        original_edit_prompt: Original user prompt for filename.
        output_dir: Optional directory to save the image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Tuple of (image bytes, saved file path).
    """
    edit_instructions = build_transformation_prompt(optimized_edit_prompt)
    return await generate_and_save_image(
        [edit_instructions, source_image], original_edit_prompt, output_dir=output_dir
    )


async def decode_base64_image(encoded_image: str) -> tuple[PIL.Image.Image, str]:
    """Decode a base64 data URL into a PIL Image.

    Args:
        encoded_image: Base64 data URL in format "data:image/[format];base64,[data]".

    Returns:
        Tuple of (PIL Image, MIME type string).

    Raises:
        ValueError: If format is invalid or image cannot be decoded.
    """
    if not encoded_image.startswith("data:image/"):
        raise ValueError(
            "Invalid image format. Expected data:image/[format];base64,[data]"
        )

    try:
        image_format, image_data = encoded_image.split(";base64,")
        image_format = image_format.replace("data:", "")
        image_bytes = base64.b64decode(image_data)
        source_image = PIL.Image.open(BytesIO(image_bytes))
        logger.info(f"Successfully loaded image with format: {image_format}")
        return source_image, image_format
    except binascii.Error as e:
        logger.error(f"Error: Invalid base64 encoding: {e!s}")
        raise ValueError(
            "Invalid base64 encoding. Please provide a valid base64 encoded image."
        ) from e
    except ValueError as e:
        logger.error(f"Error: Invalid image data format: {e!s}")
        raise ValueError(
            "Invalid image data format. "
            "Image must be in format 'data:image/[format];base64,[data]'"
        ) from e
    except PIL.UnidentifiedImageError as e:
        logger.error("Error: Could not identify image format")
        raise ValueError(
            "Could not identify image format. "
            "Supported formats include PNG, JPEG, GIF, WebP."
        ) from e


@mcp.tool()
async def generate_image_from_text(
    prompt: str, output_dir: str | None = None
) -> tuple[bytes, str]:
    """Generate an image from a text prompt using Gemini.

    Args:
        prompt: Text description of the desired image.
        output_dir: Optional directory to save the generated image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Tuple of (image bytes, saved file path).
    """
    translated_prompt = await translate_to_english(prompt)
    contents = build_generation_prompt(translated_prompt)
    return await generate_and_save_image([contents], prompt, output_dir=output_dir)


@mcp.tool()
async def transform_image_from_encoded(
    encoded_image: str, prompt: str, output_dir: str | None = None
) -> tuple[bytes, str]:
    """Transform a base64-encoded image using Gemini.

    Args:
        encoded_image: Base64 data URL (data:image/[format];base64,[data]).
        prompt: Text description of desired transformation.
        output_dir: Optional directory to save the generated image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Tuple of (image bytes, saved file path).
    """
    logger.info(
        f"Processing transform_image_from_encoded request with prompt: {prompt}"
    )

    source_image, _ = await decode_base64_image(encoded_image)
    translated_prompt = await translate_to_english(prompt)
    return await transform_and_save_image(
        source_image, translated_prompt, prompt, output_dir
    )


@mcp.tool()
async def transform_image_from_file(
    image_file_path: str, prompt: str, output_dir: str | None = None
) -> tuple[bytes, str]:
    """Transform an image file using Gemini.

    Args:
        image_file_path: Path to the source image file.
        prompt: Text description of desired transformation.
        output_dir: Optional directory to save the generated image.
                   If not provided, uses DEFAULT_OUTPUT_IMAGE_PATH environment variable.
                   If that's also not set, uses current working directory.

    Returns:
        Tuple of (image bytes, saved file path).
    """
    logger.info(f"Processing transform_image_from_file request with prompt: {prompt}")
    logger.info(f"Image file path: {image_file_path}")

    if not Path(image_file_path).exists():
        raise ValueError(f"Image file not found: {image_file_path}")

    translated_prompt = await translate_to_english(prompt)

    try:
        source_image = PIL.Image.open(image_file_path)
        logger.info(f"Successfully loaded image from file: {image_file_path}")
    except PIL.UnidentifiedImageError:
        logger.error("Error: Could not identify image format")
        raise ValueError(
            "Could not identify image format. "
            "Supported formats include PNG, JPEG, GIF, WebP."
        ) from None

    return await transform_and_save_image(
        source_image, translated_prompt, prompt, output_dir
    )


def main() -> None:
    """Start the MCP server."""
    logger.info("Starting Gemini Image Generator MCP server...")
    mcp.run(transport="stdio")
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
