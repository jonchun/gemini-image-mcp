# Gemini Image Generator MCP Server

Generate and transform images using Google's Gemini AI model through the Model Context Protocol (MCP).

## Getting Started

### 1. Get a Gemini API Key

1. Visit [Google AI Studio API Keys page](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/gemini-image-mcp.git
cd gemini-image-mcp

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Configure Claude Desktop

Add to your `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "gemini-image-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/gemini-image-mcp",
        "run",
        "gemini-image-mcp-server"
      ],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key",
        "DEFAULT_OUTPUT_IMAGE_PATH": "/ABSOLUTE/PATH/TO/images"
      }
    }
  }
}
```

Replace:

- `/ABSOLUTE/PATH/TO/gemini-image-mcp` with where you cloned this repo
- `your-gemini-api-key` with your actual Gemini API key
- `/ABSOLUTE/PATH/TO/images` with where you want images saved by default (optional - defaults to current working directory)

### 4. Restart Claude Desktop

Restart Claude Desktop to load the server.

## Usage

Ask Claude to generate or transform images:

**Generate images:**

- "Generate an image of a sunset over mountains"
- "Create a 3D rendered flying pig with a top hat over a futuristic city"

**Transform images:**

- "Add snow to this landscape"
- "Add a cute baby whale flying alongside the pig"

## Features

- **Text-to-image generation** using Gemini 3 Pro Image Preview
- **Image-to-image transformation** from text prompts
- **Automatic filename generation** based on prompts
- **High-resolution output** with strict text exclusion
- **Multiple input formats** - file paths or base64-encoded images
- **Flexible output directory** - specify per-request or use default

## Available Tools

### `generate_image_from_text`

Creates a new image from a text description.

```python
generate_image_from_text(prompt: str, output_dir: str | None = None) -> str
```

**Parameters:**

- `prompt`: Text description of the image to generate
- `output_dir` (optional): Directory to save the generated image. If not provided, uses `DEFAULT_OUTPUT_IMAGE_PATH` environment variable, or current working directory if not set.

**Returns:** Path to the saved image file

**Example:**

```
"Generate a photorealistic sunset over mountains with purple sky"
```

### `transform_image_from_file`

Transforms an existing image based on a text prompt.

```python
transform_image_from_file(image_file_path: str, prompt: str, output_dir: str | None = None) -> str
```

**Parameters:**

- `image_file_path`: Path to the image file
- `prompt`: Description of how to transform the image
- `output_dir` (optional): Directory to save the generated image. If not provided, uses `DEFAULT_OUTPUT_IMAGE_PATH` environment variable, or current working directory if not set.

**Returns:** Path to the transformed image file

**Example:**

```
transform_image_from_file("/path/to/image.png", "Add a rainbow in the sky")
```

### `transform_image_from_encoded`

Transforms an image from base64-encoded data.

```python
transform_image_from_encoded(encoded_image: str, prompt: str, output_dir: str | None = None) -> str
```

**Parameters:**

- `encoded_image`: Base64 encoded image with format header (`data:image/[format];base64,[data]`)
- `prompt`: Description of how to transform the image
- `output_dir` (optional): Directory to save the generated image. If not provided, uses `DEFAULT_OUTPUT_IMAGE_PATH` environment variable, or current working directory if not set.

**Returns:** Path to the transformed image file

**Note:** This method may be slower due to base64 encoding overhead.

## Configuration

Environment variables:

- `GEMINI_API_KEY` (required): Your Gemini API key
- `DEFAULT_OUTPUT_IMAGE_PATH` (optional): Default directory to save generated images (default: current working directory)
- `GEMINI_MODEL` (optional): Model to use (default: `gemini-2.5-flash-image`)
- `GEMINI_BASE_URL` (optional): API base URL (default: `https://generativelanguage.googleapis.com`)

You can also specify the output directory per-request by passing the `output_dir` parameter to any of the image generation tools.

## Development

Test the server using FastMCP development mode:

```bash
fastmcp dev server.py
```

This starts a local server at http://localhost:5173/ with the MCP Inspector for testing tools directly.

## Requirements

- Python 3.11+
- Google Gemini API key
- MCP-compatible client (Claude Desktop, Cursor, etc.)

## License

MIT License
