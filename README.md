# 3D Model Generator with RAG

This project uses Retrieval Augmented Generation (RAG) to create 3D models in OpenSCAD based on text descriptions. It supports multiple LLM providers including Claude-3-Sonnet, O1-Mini, Gemma3 4B, and DeepSeek-R1 7B.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API configuration:
```bash
API_KEY=your_api_key_here
BASE_URL=your_base_url_here
```

3. For Local Models (optional):
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull the required models:
     ```bash
     # For Gemma3 4B
     ollama pull gemma3:4b-it-q8_0
     
     # For DeepSeek-R1 7B
     ollama pull deepseek-r1:7b
     ```
   - Start the Ollama service

4. Install OpenSCAD from [https://openscad.org/downloads.html](https://openscad.org/downloads.html)

## Usage

1. Run the script:
```bash
python rag_3d_modeler.py
```

2. Choose from the main menu options:
   - 1: Generate a 3D object
   - 2: Input knowledge manually
   - 3: Delete knowledge
   - 4: Quit

### Generating 3D Objects (Option 1)
If you select option 1 (Generate a 3D object):
1. Select your preferred LLM provider:
   - 1: Anthropic (Claude-3-Sonnet) - Default, best for complex designs
   - 2: OpenAI (O1-Mini) - Lightweight alternative
   - 3: Gemma3 4B - Local, efficient model
   - 4: DeepSeek-R1 7B - Local model with advanced reasoning
2. Enter a description of the 3D object you want to create
3. The script will generate OpenSCAD code and save it to `output.scad`
4. A debug.txt file will be created with the full model response and reasoning process
5. You'll have the option to save the generated model to the knowledge base

### Managing Knowledge Base
The system maintains a knowledge base of 3D models that can be used to improve future generations.

#### Adding Knowledge Manually (Option 2)
1. Enter a description/query for the 3D model
2. Input the OpenSCAD code (press Enter twice to finish)
3. The knowledge will be automatically saved with a filename based on the main object (e.g., "cup1.pkl")

#### Deleting Knowledge (Option 3)
1. Enter the name of the knowledge file to delete (e.g., "snowman1" for snowman1.pkl)
2. Confirm the deletion when prompted

## Examples

Try these example prompts:
- "Create a simple cup with a handle"
- "Make a basic chair with a backrest"
- "Generate a hexagonal pencil holder"

## How it Works

1. The system uses a knowledge base of OpenSCAD fundamentals
2. Your description is processed using the selected LLM provider
3. The system generates appropriate OpenSCAD code
4. The code is saved to output.scad and debug information to debug.txt
5. The generated code can be opened in OpenSCAD

## LLM Provider Comparison

### Anthropic (Claude-3-Sonnet)
- Best for complex designs
- Excellent understanding of 3D modeling concepts
- Default choice for production use
- Requires API key
- Most reliable output format

### OpenAI (O1-Mini)
- Lightweight model
- Good for basic shapes
- Faster response times
- Requires API key
- Includes reasoning process in output

### Gemma3 4B
- Runs completely locally
- No API key required
- 4B parameter model
- Good balance of performance and resource usage
- Efficient for simple to moderate designs
- Requires GPU for optimal performance

### DeepSeek-R1 7B
- Runs completely locally
- Advanced reasoning capabilities
- 7B parameter model
- Excellent for complex designs
- Includes detailed reasoning process
- Higher resource requirements
- No API key required
- Best local model for complex designs

## Debug Information

The system creates two main output files:
1. `output.scad`: Contains the clean, generated OpenSCAD code
2. `debug.txt`: Contains:
   - The LLM provider used
   - The full prompt sent to the model
   - The raw model response
   - Parsed components (code and reasoning)
   - Any errors that occurred

## Limitations

- The system generates basic 3D models. Complex designs may need manual refinement
- All measurements are approximate and may need adjustment
- Some complex shapes may require manual optimization
- Local models require sufficient computational resources:
  - Gemma3 4B: Minimum 8GB RAM
  - DeepSeek-R1 7B: Recommended 16GB RAM
- API-based models require valid API keys and internet connection