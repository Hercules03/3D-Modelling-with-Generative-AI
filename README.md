# 3D Modelling with Generative AI

A Python-based system that uses Large Language Models (LLMs) to generate OpenSCAD code for 3D models from textual descriptions. Built with a focus on maintainability and extensibility, the system leverages modern AI techniques including:

- **Retrieval-Augmented Generation (RAG)** for accessing relevant examples
- **Semantic vector search** for finding similar models
- **Step-back analysis** for technical understanding of 3D objects 
- **Comprehensive metadata extraction** for improved model generation

## Core Features

### Advanced LLM Integration
- **Multi-provider support**: Anthropic Claude, OpenAI, Ollama (local)
- **Configurable generation parameters**: Control temperature, context window
- **Prompt optimization**: Carefully designed prompts for consistent results

### Intelligent Analysis Pipeline
- **Keyword Extraction**: Accurately identifies object types and modifiers
- **Step-Back Analysis**: Breaks down design requirements into:
  - Core principles (design philosophy)
  - Shape components (geometric building blocks)
  - Implementation steps (construction approach)
- **Metadata Extraction**: Comprehensive analysis of object properties

### Knowledge Management
- **ChromaDB Vector Store**: Semantic search for finding relevant examples
- **Automatic Categorization**: Smart classification of objects
- **Example Management**: Add, view, and delete examples with metadata
- **Smart Deduplication**: Prevents redundant examples 

### Enhanced Result Ranking
- **Multi-factor scoring**: Weighted matching based on:
  - Component similarity (35%)
  - Geometric properties (25%)
  - Step-back analysis (20%)
  - Feature matching (15%)
  - Style and complexity (5%)

### Robust Architecture
- **Modular Design**: Separate components with clear responsibilities
- **Comprehensive Logging**: Debug and trace entire generation process
- **Error Handling**: Graceful handling of API and processing failures

## System Components

The system is built with a modular design pattern for maintainability:

- **`rag_3d_modeler.py`**: Main application interface
- **`OpenSCAD_Generator.py`**: Core generation logic
- **`enhanced_scad_knowledge_base.py`**: Knowledge base management
- **`step_back_analyzer.py`**: Technical analysis of 3D objects
- **`metadata_extractor.py`**: Comprehensive metadata extraction
- **`KeywordExtractor.py`**: Object type and modifier identification
- **`llm_management.py`**: Provider management and optimization
- **`conversation_logger.py`**: Interaction tracking
- **`prompts.py`**: System prompts and templates

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/3D-Modelling-with-Generative-AI.git
cd 3D-Modelling-with-Generative-AI
```

2. **Set up a virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API keys**:
Create a `.env` file in the project root with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

5. **(Optional) Install Ollama for local model support**:
```bash
# Install Ollama from https://ollama.ai/
# Then pull required models:
ollama pull gemma3:4b-it-q8_0
ollama pull deepseek-r1:7b
```

## Usage

Run the main application:
```bash
python rag_3d_modeler.py
```

### Workflow

1. **Generate a 3D object**:
   - Enter a description of the object you want to model
   - Review and approve keyword extraction
   - Confirm step-back technical analysis
   - System retrieves and ranks relevant examples
   - Review the generated OpenSCAD code
   - Save and render the model

2. **Knowledge Management**:
   - View existing examples in the knowledge base
   - Add new examples manually
   - Delete or update examples
   - Browse categorized examples

## Example Session

```
Welcome to the 3D Model Generator!

What would you like to model? A coffee mug with a hexagonal pattern and a curved handle

=== KEYWORD EXTRACTION ===
Analyzing description...

Keyword Analysis Results:
Core Type: mug
Modifiers: [coffee, hexagonal, pattern, curved, handle]
Compound Type: coffee mug

Do you accept these keywords? (yes/no): yes

=== TECHNICAL ANALYSIS ===
Performing step-back analysis...

Core Principles:
- Cylindrical container with appropriate dimensions for fluid containment
- Ergonomic handle design with curved form for comfortable grip
- Hexagonal tessellation pattern as a decorative and structural element
- Balance between aesthetics and functional usability

Shape Components:
- Cylindrical body with hollow interior
- Curved handle with attachment points
- Hexagonal pattern on exterior surface
- Solid base for stability

Implementation Steps:
1. Create cylindrical container with appropriate dimensions
2. Design curved handle and position it relative to the mug body
3. Generate hexagonal pattern and apply it to the mug's exterior
4. Add finishing details like base reinforcement and rim
5. Combine all elements with proper boolean operations

Do you accept this technical analysis? (yes/no): yes

=== FINDING SIMILAR EXAMPLES ===
Searching knowledge base...
Found 2 relevant examples

=== GENERATING SCAD CODE ===
Generating OpenSCAD code...

Generated Code:
// Coffee Mug with Hexagonal Pattern and Curved Handle
// Parameters
$fn = 100; // Smoothness
mug_height = 95; // Height of mug
mug_radius = 40; // Radius of mug
wall_thickness = 3; // Thickness of the mug wall
handle_thickness = 7; // Thickness of the handle
...

Would you like to save this example to the knowledge base? (y/n): y
Example added to knowledge base!
```

## Requirements

- Python 3.8+
- OpenSCAD (for rendering models)
- Dependencies listed in `requirements.txt`

## Recent Improvements

- **Enhanced Metadata Extraction**: Consolidated metadata extraction into a single class
- **Step-Back Analyzer**: Dedicated module for technical analysis of 3D objects
- **Improved Error Handling**: Better handling of edge cases and API failures
- **Code Organization**: Refactored code for better modularity and maintainability
- **Performance Optimization**: Improved vector search and ranking algorithms

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenSCAD community
- Anthropic and OpenAI for their LLM APIs
- Ollama project for local model support
- ChromaDB team for their vector database implementation