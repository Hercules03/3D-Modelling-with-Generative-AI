# 3D Model Generator with RAG

This project uses Retrieval Augmented Generation (RAG) to create 3D models in OpenSCAD based on text descriptions. It supports multiple LLM providers including Claude-3-Sonnet, O1-Mini, and Gemma3 4B via Ollama.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API configuration:
```bash
API_KEY=your_api_key_here
```

3. For Ollama (optional):
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the Gemma3 4B model:
     ```bash
     ollama pull gemma3:4b
     ```
   - Start the Ollama service

4. Install OpenSCAD from [https://openscad.org/downloads.html](https://openscad.org/downloads.html)

## Usage

1. Run the script:
```bash
python rag_3d_modeler.py
```

2. Select your preferred LLM provider:
   - 1: Anthropic (Claude-3-Sonnet) - Default, best for complex designs
   - 2: OpenAI (O1-Mini) - Lightweight alternative
   - 3: Ollama (Gemma3 4B) - Local, open-source model

3. Enter a description of the 3D object you want to create.

4. The script will generate OpenSCAD code and save it to `output.scad`.

5. Open the generated `output.scad` file in OpenSCAD to view and export your 3D model.

## Examples

Try these example prompts:
- "Create a simple cup with a handle"
- "Make a basic chair with a backrest"
- "Generate a hexagonal pencil holder"

## How it Works

1. The system uses a knowledge base of OpenSCAD fundamentals
2. Your description is processed using the selected LLM provider
3. The system generates appropriate OpenSCAD code
4. The code is saved to a file that can be opened in OpenSCAD

## LLM Provider Comparison

### Anthropic (Claude-3-Sonnet)
- Best for complex designs
- Excellent understanding of 3D modeling concepts
- Default choice
- Requires API key

### OpenAI (O1-Mini)
- Lightweight model
- Good for basic shapes
- Faster response times
- Requires API key

### Ollama (Gemma3 4B)
- Runs completely locally
- No API key required
- 4B parameter model
- Good balance of performance and resource usage
- Requires GPU for optimal performance

## Limitations

- The system generates basic 3D models. Complex designs may need manual refinement
- All measurements are approximate and may need adjustment
- Some complex shapes may require manual optimization
- Local models (Gemma3) require sufficient computational resources