# 3D Modelling with Generative AI

A Python-based tool that uses various Large Language Models (LLMs) to generate OpenSCAD code for 3D models based on text descriptions. The system employs intelligent example validation, ChromaDB-powered semantic search, and maintains a curated knowledge base to improve generation quality.

## Features

- **Multiple LLM Provider Support**:
  - Anthropic (Claude-3-Sonnet)
  - OpenAI (O1-Mini)
  - Gemma (via Ollama)
  - DeepSeek (via Ollama)

- **Enhanced Knowledge Base with ChromaDB**:
  - Vector-based semantic search for better example matching
  - Persistent storage of examples with automatic deduplication
  - Multi-factor ranking system combining:
    - Semantic similarity (25%)
    - Step-back analysis matching (25%)
    - Component matching (25%)
    - Metadata matching (15%)
    - Complexity scoring (10%)
  - Configurable similarity thresholds
  - Efficient incremental updates

- **Smart Example Management**:
  - Advanced component-based filtering
  - Sophisticated complexity scoring system
  - Automatic metadata extraction
  - Clear validation feedback in debug logs

- **Category and Property System**:
  - Hierarchical categorization of objects
  - Standardized properties for filtering
  - Multi-level category matching
  - Property-based refinement
  - Similar object suggestions

- **Step-Back Analysis Integration**:
  - Structured parsing of technical requirements
  - Core principles extraction
  - Shape component analysis
  - Implementation step planning
  - Enhanced example matching

- **Debugging Support**:
  - Comprehensive debug logs in `debug.txt`
  - Detailed similarity score breakdowns
  - Component matching analysis
  - Complexity score calculations
  - Tracks API interactions and validation decisions

## Prerequisites

1. Python 3.x
2. OpenSCAD installed on your system
3. API keys for Anthropic and/or OpenAI (if using those providers)
4. Ollama installed (required for example validation and optional LLM providers)

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

4. Install Ollama and required models:
```bash
# Required for example validation
ollama pull gemma3:1b

# Optional for generation (if using these providers)
ollama pull gemma3:4b-it-q8_0
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
3. System retrieves and validates relevant examples
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

Retrieving knowledge from SCAD knowledge base...
Searching for examples related to: chair

Validating usefulness of 3 examples using Gemma3 1B...
[Example validation details will be shown here]
```

## Project Structure

- `rag_3d_modeler.py`: Main application file
- `enhanced_scad_knowledge_base.py`: Enhanced knowledge base with ChromaDB integration
- `OpenSCAD_Generator.py`: Core generation logic
- `LLM.py`: LLM provider management
- `ExampleValidator.py`: Intelligent example validation
- `prompts.py`: System prompts and templates
- `constant.py`: System constants and configurations

## Advanced Features

### Complexity Scoring
The system uses a sophisticated scoring mechanism that evaluates:
- Code structure (70%):
  - Operations and functions
  - Nesting levels
  - Variables and modules
  - Geometric operations
- Metadata attributes (30%):
  - Declared complexity level
  - Printability assessment
  - Support requirements

### Component-Based Filtering
Implements intelligent component matching:
- Extracts components from queries and examples
- Analyzes geometric primitives and operations
- Matches based on technical requirements
- Provides detailed component compatibility scores

### Result Ranking
Multi-factor ranking system combining:
- Base semantic similarity
- Step-back analysis matching
- Component compatibility
- Metadata alignment
- Complexity appropriateness

## Debug Information

The system now provides enhanced debugging information including:
- Similarity score breakdowns
- Component matching details
- Complexity calculations
- Step-back analysis results
- ChromaDB query analytics

### Example Debug Output
```
=== SEARCH RESULTS ANALYSIS ===
Query: "Complex coffee mug with decorative handle"
--------------------------------------------------
Top Match Scores:
- Final Score: 0.85
  - Semantic Similarity: 0.92
  - Step-back Match: 0.88
  - Component Match: 0.85
  - Metadata Match: 0.78
  - Complexity Score: 0.70
--------------------------------------------------
Component Analysis:
- Required: ['cylinder', 'handle', 'boolean_ops']
- Found: ['cylinder', 'handle', 'difference']
- Match Rate: 100%
==================================================
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Your chosen license]

## Acknowledgments

- OpenSCAD community
- LLM providers (Anthropic, OpenAI, Google, DeepSeek)
- Ollama project