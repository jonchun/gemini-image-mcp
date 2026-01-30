import asyncio
import base64
import binascii
import contextlib
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
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent

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
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")


async def invoke_gemini_api(
    contents: list[Any],
    model: str | None = None,
    config: types.GenerateContentConfig | None = None,
    text_only: bool = False,
) -> str | tuple[bytes, str]:
    """Call the Gemini API and return text or image data.

    Args:
        contents: Content to send (text and/or images).
        model: Gemini model name. Defaults to GEMINI_MODEL.
        config: Optional generation config.
        text_only: If True, return text response; otherwise return image bytes with mime type.

    Returns:
        Text string if text_only=True, otherwise tuple of (raw image bytes, mime_type).

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
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e!s}")
        raise

    logger.info(f"Response received from Gemini API using model {model}")

    if text_only:
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            raise ValueError("No text content found in Gemini response")
        text = response.candidates[0].content.parts[0].text
        if text is None:
            raise ValueError("No text content found in Gemini response")
        return text.strip()

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None and part.inline_data.data is not None:
                mime_type = part.inline_data.mime_type or "image/png"
                logger.info(f"Image received with mime_type: {mime_type}")
                return part.inline_data.data, mime_type

    raise ValueError("No image data found in Gemini response")


async def invoke_gemini_api_with_progress(
    contents: list[Any],
    ctx: Context | None,
    model: str | None = None,
    config: types.GenerateContentConfig | None = None,
    text_only: bool = False,
    progress_start: float = 0.2,
    progress_end: float = 0.8,
) -> str | tuple[bytes, str]:
    """Call Gemini API while sending periodic progress updates.

    Args:
        contents: Content to send (text and/or images).
        ctx: Context for progress reporting.
        model: Gemini model name. Defaults to GEMINI_MODEL.
        config: Optional generation config.
        text_only: If True, return text response; otherwise return image bytes with mime type.
        progress_start: Starting progress value (0.0-1.0).
        progress_end: Ending progress value (0.0-1.0).

    Returns:
        Text string if text_only=True, otherwise tuple of (raw image bytes, mime_type).
    """

    async def send_periodic_progress() -> None:
        """Background task to send progress updates every 10 seconds."""
        progress = progress_start
        increment = (progress_end - progress_start) / 12  # ~2 min max / 10 sec intervals
        while True:
            await asyncio.sleep(10)
            progress = min(progress + increment, progress_end)
            if ctx:
                await ctx.report_progress(progress, message="Generating image...")

    # Start heartbeat task if context is provided
    heartbeat_task = None
    if ctx:
        heartbeat_task = asyncio.create_task(send_periodic_progress())

    try:
        # Make the actual API call
        result = await invoke_gemini_api(contents, model, config, text_only)
        return result
    finally:
        # Cancel heartbeat when done
        if heartbeat_task:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task


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


async def generate_image(
    contents: list[Any],
    model: str | None = None,
) -> tuple[bytes, str]:
    """Generate an image with Gemini.

    Args:
        contents: Content list for Gemini (prompt and optional source image).
        model: Optional Gemini model override.

    Returns:
        Tuple of (raw image bytes, mime_type).
    """
    gemini_response = await invoke_gemini_api(
        contents,
        model=model,
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )
    assert isinstance(gemini_response, tuple), "Expected tuple response"
    return gemini_response


async def generate_image_with_progress(
    contents: list[Any],
    ctx: Context | None,
    model: str | None = None,
) -> tuple[bytes, str]:
    """Generate an image with Gemini, with progress reporting.

    Args:
        contents: Content list for Gemini (prompt and optional source image).
        ctx: Context for progress reporting.
        model: Optional Gemini model override.

    Returns:
        Tuple of (raw image bytes, mime_type).
    """
    # Send periodic progress during Gemini API call (20% -> 80%)
    gemini_response = await invoke_gemini_api_with_progress(
        contents,
        ctx,
        model=model,
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        progress_start=0.2,
        progress_end=0.8,
    )
    assert isinstance(gemini_response, tuple), "Expected tuple response"
    return gemini_response


async def transform_image(
    source_image: PIL.Image.Image,
    optimized_edit_prompt: str,
    ctx: Context | None = None,
    model: str | None = None,
) -> tuple[bytes, str]:
    """Transform an image with Gemini.

    Args:
        source_image: PIL Image to transform.
        optimized_edit_prompt: Translated/optimized transformation prompt.
        ctx: Optional context for progress reporting.
        model: Optional Gemini model override.

    Returns:
        Tuple of (raw image bytes, mime_type).
    """
    edit_instructions = build_transformation_prompt(optimized_edit_prompt)
    return await generate_image_with_progress([edit_instructions, source_image], ctx, model=model)


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
        raise ValueError("Invalid image format. Expected data:image/[format];base64,[data]")

    try:
        image_format, image_data = encoded_image.split(";base64,")
        image_format = image_format.replace("data:", "")
        image_bytes = base64.b64decode(image_data)
        source_image = PIL.Image.open(BytesIO(image_bytes))
        logger.info(f"Successfully loaded image with format: {image_format}")
        return source_image, image_format
    except binascii.Error as e:
        logger.error(f"Error: Invalid base64 encoding: {e!s}")
        raise ValueError("Invalid base64 encoding. Please provide a valid base64 encoded image.") from e
    except ValueError as e:
        logger.error(f"Error: Invalid image data format: {e!s}")
        raise ValueError(
            "Invalid image data format. Image must be in format 'data:image/[format];base64,[data]'"
        ) from e
    except PIL.UnidentifiedImageError as e:
        logger.error("Error: Could not identify image format")
        raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.") from e


@mcp.tool()
async def generate_image_from_text(
    prompt: str, output_dir: str | None = None, model: str | None = None, ctx: Context | None = None
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt using Gemini.

    Args:
        prompt: Text description of the desired image.
        output_dir: Optional directory to save the generated image.
                   If not provided, the image is only returned in the response (not saved to disk).
        model: Optional Gemini model name. If not provided, uses GEMINI_MODEL environment variable.
        ctx: Optional context for progress reporting.

    Returns:
        List containing ImageContent with the generated image, and optionally TextContent with file path if saved.
    """
    # Progress: Starting
    if ctx:
        await ctx.report_progress(0.05, message="Starting image generation...")

    # Progress: Translating prompt
    if ctx:
        await ctx.report_progress(0.1, message="Translating prompt...")
    translated_prompt = await translate_to_english(prompt)

    # Progress: Preparing generation
    if ctx:
        await ctx.report_progress(0.15, message="Preparing image generation...")
    contents = build_generation_prompt(translated_prompt)

    # Generate with progress reporting (20% -> 80%)
    image_bytes, mime_type = await generate_image_with_progress([contents], ctx, model=model)

    # Build response
    result: list[TextContent | ImageContent] = []

    # Only save to disk if output_dir is explicitly provided
    if output_dir is not None:
        if ctx:
            await ctx.report_progress(0.85, message="Generating filename...")
        filename = await generate_filename_from_prompt(prompt)

        if ctx:
            await ctx.report_progress(0.95, message="Saving image to disk...")
        saved_path = await save_image_to_disk(image_bytes, filename, output_dir)
        result.append(TextContent(type="text", text=f"Image saved to: {saved_path}"))

    # Progress: Complete
    if ctx:
        await ctx.report_progress(1.0, message="Image generation complete!")

    # Always include the image content with actual mime type from Gemini
    result.append(ImageContent(type="image", data=base64.b64encode(image_bytes).decode("utf-8"), mimeType=mime_type))

    return result


@mcp.tool()
async def transform_image_from_encoded(
    encoded_image: str, prompt: str, output_dir: str | None = None, model: str | None = None, ctx: Context | None = None
) -> list[TextContent | ImageContent]:
    """Transform a base64-encoded image using Gemini.

    Args:
        encoded_image: Base64 data URL (data:image/[format];base64,[data]).
        prompt: Text description of desired transformation.
        output_dir: Optional directory to save the generated image.
                   If not provided, the image is only returned in the response (not saved to disk).
        model: Optional Gemini model name. If not provided, uses GEMINI_MODEL environment variable.
        ctx: Optional context for progress reporting.

    Returns:
        List containing ImageContent with the transformed image, and optionally TextContent with file path if saved.
    """
    logger.info(f"Processing transform_image_from_encoded request with prompt: {prompt}")

    # Progress: Starting
    if ctx:
        await ctx.report_progress(0.05, message="Starting image transformation...")

    # Progress: Decoding image
    if ctx:
        await ctx.report_progress(0.08, message="Decoding image...")
    source_image, _ = await decode_base64_image(encoded_image)

    # Progress: Translating prompt
    if ctx:
        await ctx.report_progress(0.1, message="Translating prompt...")
    translated_prompt = await translate_to_english(prompt)

    # Progress: Preparing transformation
    if ctx:
        await ctx.report_progress(0.15, message="Preparing image transformation...")

    # Transform with progress reporting (20% -> 80%)
    image_bytes, mime_type = await transform_image(source_image, translated_prompt, ctx, model=model)

    # Build response
    result: list[TextContent | ImageContent] = []

    # Only save to disk if output_dir is explicitly provided
    if output_dir is not None:
        if ctx:
            await ctx.report_progress(0.85, message="Generating filename...")
        filename = await generate_filename_from_prompt(prompt)

        if ctx:
            await ctx.report_progress(0.95, message="Saving image to disk...")
        saved_path = await save_image_to_disk(image_bytes, filename, output_dir)
        result.append(TextContent(type="text", text=f"Image saved to: {saved_path}"))

    # Progress: Complete
    if ctx:
        await ctx.report_progress(1.0, message="Image transformation complete!")

    # Always include the image content with actual mime type from Gemini
    result.append(ImageContent(type="image", data=base64.b64encode(image_bytes).decode("utf-8"), mimeType=mime_type))

    return result


@mcp.tool()
async def transform_image_from_file(
    image_file_path: str,
    prompt: str,
    output_dir: str | None = None,
    model: str | None = None,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Transform an image file using Gemini.

    Args:
        image_file_path: Path to the source image file.
        prompt: Text description of desired transformation.
        output_dir: Optional directory to save the generated image.
                   If not provided, the image is only returned in the response (not saved to disk).
        model: Optional Gemini model name. If not provided, uses GEMINI_MODEL environment variable.
        ctx: Optional context for progress reporting.

    Returns:
        List containing ImageContent with the transformed image, and optionally TextContent with file path if saved.
    """
    logger.info(f"Processing transform_image_from_file request with prompt: {prompt}")
    logger.info(f"Image file path: {image_file_path}")

    # Progress: Starting
    if ctx:
        await ctx.report_progress(0.05, message="Starting image transformation...")

    # Progress: Loading image file
    if ctx:
        await ctx.report_progress(0.08, message="Loading image file...")

    if not Path(image_file_path).exists():
        raise ValueError(f"Image file not found: {image_file_path}")

    try:
        source_image = PIL.Image.open(image_file_path)
        logger.info(f"Successfully loaded image from file: {image_file_path}")
    except PIL.UnidentifiedImageError:
        logger.error("Error: Could not identify image format")
        raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.") from None

    # Progress: Translating prompt
    if ctx:
        await ctx.report_progress(0.1, message="Translating prompt...")
    translated_prompt = await translate_to_english(prompt)

    # Progress: Preparing transformation
    if ctx:
        await ctx.report_progress(0.15, message="Preparing image transformation...")

    # Transform with progress reporting (20% -> 80%)
    image_bytes, mime_type = await transform_image(source_image, translated_prompt, ctx, model=model)

    # Build response
    result: list[TextContent | ImageContent] = []

    # Only save to disk if output_dir is explicitly provided
    if output_dir is not None:
        if ctx:
            await ctx.report_progress(0.85, message="Generating filename...")
        filename = await generate_filename_from_prompt(prompt)

        if ctx:
            await ctx.report_progress(0.95, message="Saving image to disk...")
        saved_path = await save_image_to_disk(image_bytes, filename, output_dir)
        result.append(TextContent(type="text", text=f"Image saved to: {saved_path}"))

    # Progress: Complete
    if ctx:
        await ctx.report_progress(1.0, message="Image transformation complete!")

    # Always include the image content with actual mime type from Gemini
    result.append(ImageContent(type="image", data=base64.b64encode(image_bytes).decode("utf-8"), mimeType=mime_type))

    return result


def main() -> None:
    """Start the MCP server."""
    logger.info("Starting Gemini Image Generator MCP server...")
    mcp.run(transport="stdio")
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
