# 3D Modelling with Generative AI

A Python-based tool that uses various Large Language Models (LLMs) to generate OpenSCAD code for 3D models based on text descriptions. The system employs a step-back prompting approach and maintains a knowledge base of examples to improve generation quality.

## Features

- **Multiple LLM Provider Support**:
  - Anthropic (Claude-3-Sonnet)
  - OpenAI (O1-Mini)
  - Gemma (via Ollama)
  - DeepSeek (via Ollama)

- **Step-back Analysis**: Performs geometric and technical analysis before code generation to ensure better understanding of the requirements.

- **Knowledge Base Management**:
  - Stores successful examples for future reference
  - Retrieves relevant examples to improve generation
  - Allows manual input of new examples
  - Supports deletion of stored examples

- **Debugging Support**:
  - Comprehensive debug logs in `debug.txt`
  - Tracks API interactions
  - Records model responses and parsing steps
  - Monitors error conditions

## Prerequisites

1. Python 3.x
2. OpenSCAD installed on your system
3. API keys for Anthropic and/or OpenAI (if using those providers)
4. Ollama installed (if using Gemma or DeepSeek)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3D-Modelling-with-Generative-AI.git
cd 3D-Modelling-with-Generative-AI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key
```

4. (Optional) For Gemma or DeepSeek, install Ollama and pull the models:
```bash
# For Gemma
ollama pull gemma3:4b-it-q8_0

# For DeepSeek
ollama pull deepseek-r1:7b
```

## Usage

Run the main script:
```bash
python rag_3d_modeler.py
```

### Options:
1. Generate a 3D object
2. Input knowledge manually
3. Delete knowledge
4. Quit

When generating a 3D object:
1. Select your preferred LLM provider
2. Enter your object description
3. Review and validate the step-back analysis
4. Review the generated OpenSCAD code
5. Optionally save successful generations to the knowledge base

### Example:
```bash
$ python rag_3d_modeler.py
Welcome to the 3D Model Generator!

Select an option:
1. Generate a 3D object
2. Input knowledge manually
3. Delete knowledge
4. Quit

Enter your choice (1-4): 1

Available LLM Providers:
1. Anthropic (Claude-3-Sonnet)
2. OpenAI (O1-Mini)
3. Gemma3:4B
4. DeepSeek-R1:7B

Select LLM provider (1-4, default is 1): 1

What would you like to model? A simple chair
```

## Project Structure

- `rag_3d_modeler.py`: Main application file
- `OpenSCAD_Generator.py`: Core generation logic
- `LLM.py`: LLM provider management
- `SCADKnowledgeBase.py`: Knowledge base management
- `KeywordExtractor.py`: Keyword extraction for validation
- `prompts.py`: System prompts and templates
- `constant.py`: System constants and configurations

## Debug Information

The system generates detailed debug information in `debug.txt`, including:
- Provider and model information
- Full prompts sent to LLMs
- Raw responses received
- Parsed components
- Error details (if any)

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Your chosen license]

## Acknowledgments

- OpenSCAD community
- LLM providers (Anthropic, OpenAI, Google, DeepSeek)
- Ollama project