![Cover](docs/gemini_mcp_server_cover.png)

# Gemini Image Generator MCP Server

Generate and transform images using Google's Gemini AI through the Model Context Protocol (MCP).

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Text-to-Image Generation** - Create images from natural language prompts
- **Image Transformation** - Modify existing images with text descriptions
- **Automatic Filename Generation** - Smart naming based on prompts
- **Multi-Language Support** - Automatic prompt translation to English

## Installation

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/jonchun/gemini-image-mcp.git
cd gemini-image-mcp

# Using uv (recommended)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Configuration

<details>
<summary><b>Claude Desktop</b></summary>

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-image-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/gemini-image-mcp",
        "run",
        "gemini-image-mcp"
      ],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "DEFAULT_OUTPUT_IMAGE_PATH": "/path/to/images"
      }
    }
  }
}
```

</details>

<details>
<summary><b>OpenCode</b></summary>

```json
{
  "gemini-image-mcp": {
    "command": "uv",
    "args": [
      "--directory",
      "/ABSOLUTE/PATH/TO/gemini-image-mcp",
      "run",
      "gemini-image-mcp"
    ],
    "env": {
      "GEMINI_API_KEY": "your-api-key-here",
      "DEFAULT_OUTPUT_IMAGE_PATH": "/path/to/images"
    }
  }
}
```

</details>

<details>
<summary><b>Smithery</b></summary>

Install from [smithery.ai](https://smithery.ai) - search for "gemini-image-mcp".

</details>

## Usage

### Generate Images

```
Generate a photorealistic sunset over mountains with purple sky
```

![Sunset example](docs/photorealistic_sunset_mountains_purple_sky.png)

```
Create a British Shorthair silver tabby kitten playing with a ball of yarn
```

![Kitten example](docs/silver_tabby_kitten_playing_yarn.png)

### Transform Images

```
Add a cozy fireplace in the background
```

```
Add a small butterfly landing on the cat's nose
```

## Available Tools

### `generate_image_from_text`

Creates an image from a text description.

**Parameters:**

- `prompt` (required): Text description of the image
- `output_dir` (optional): Directory to save the image

**Returns:** Path to the saved image file

### `transform_image_from_file`

Transforms an existing image based on a text prompt.

**Parameters:**

- `image_file_path` (required): Path to the source image
- `prompt` (required): Description of the transformation
- `output_dir` (optional): Directory to save the image

**Returns:** Path to the transformed image file

### `transform_image_from_encoded`

Transforms a base64-encoded image.

**Parameters:**

- `encoded_image` (required): Base64 data URL (`data:image/[format];base64,[data]`)
- `prompt` (required): Description of the transformation
- `output_dir` (optional): Directory to save the image

**Returns:** Path to the transformed image file

## Configuration

| Variable                    | Required | Default                                     | Description           |
| --------------------------- | -------- | ------------------------------------------- | --------------------- |
| `GEMINI_API_KEY`            | Yes      | -                                           | Your Gemini API key   |
| `DEFAULT_OUTPUT_IMAGE_PATH` | No       | Current directory                           | Default save location |
| `GEMINI_MODEL`              | No       | `gemini-2.5-flash-image`                    | Model to use          |
| `GEMINI_BASE_URL`           | No       | `https://generativelanguage.googleapis.com` | API base URL          |

## Development

Test the server locally:

```bash
fastmcp dev src/gemini_image_mcp/server.py
```

Opens MCP Inspector at `http://localhost:5173/`

## License

MIT
