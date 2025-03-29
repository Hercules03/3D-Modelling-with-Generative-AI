# 3D Modelling with Generative AI

An intelligent Python-based system that leverages Large Language Models (LLMs) to generate OpenSCAD code for 3D models based on textual descriptions. The system employs Retrieval-Augmented Generation (RAG), semantic vector search, step-back analysis, and a comprehensive knowledge base to create high-quality 3D models.

## Key Features

### Multi-Provider LLM Support
- **Anthropic Claude-3-Sonnet**: Primary model for high-quality generation
- **OpenAI O1-Mini**: Alternative provider with optimized performance
- **Ollama Integration**: Local models support through Gemma and DeepSeek

### Intelligent Generation Process
- **Step-Back Analysis**: Deep technical analysis of requirements before generation
- **Component-Based Understanding**: Breaks down objects into fundamental components
- **Keyword Extraction**: Automated identification of core object types and modifiers
- **Multi-Stage Validation**: User confirmation at critical steps in the workflow

### Vector-Based Knowledge Management
- **ChromaDB Integration**: Semantic search for finding relevant example models
- **Auto-Deduplication**: Smart detection and handling of duplicate examples
- **Progressive Learning**: System improves with each successful generation
- **Intelligent Example Selection**: Weighted matching based on multiple factors

### Smart Ranking System
- **Multi-Factor Scoring**: Combines multiple metrics to find the most relevant examples
  - Semantic similarity (25%)
  - Step-back analysis matching (25%)
  - Component matching (25%)
  - Metadata matching (15%)
  - Complexity scoring (10%)
- **Code Complexity Analysis**: Evaluates nesting levels, operations used, and more
- **Geometric Property Matching**: Matches models based on spatial characteristics

### Comprehensive Debugging
- **Detailed Logging**: Extensive debug logs in multiple formats
- **Similarity Score Breakdowns**: Complete analysis of example matching
- **API Interaction Tracking**: Records communication with LLM providers
- **Validation Decision Tracking**: Documents the quality assessment process

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Hercules03/3D-Modelling-with-Generative-AI.git
cd 3D-Modelling-with-Generative-AI
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys**:
Create a `.env` file in the project root with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

4. **Install Ollama and required models** (for example validation and optional providers):
```bash
# Install Ollama from https://ollama.ai/
# Then pull required models:

# Required for example validation
ollama pull gemma3:1b

# Optional for generation (if using these providers)
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
   - Select your preferred LLM provider
   - Describe the object you want to model
   - Review and approve keyword extraction
   - Confirm step-back technical analysis
   - System retrieves and ranks relevant examples
   - Review the generated OpenSCAD code
   - Optionally save to knowledge base

2. **Knowledge Management**:
   - Add examples manually
   - Delete outdated examples
   - Browse the knowledge base
   - Filter by categories, complexity, and features

### Example Session

```
$ python rag_3d_modeler.py
Welcome to the 3D Model Generator!

Select an option:
1. Generate a 3D object
2. Input knowledge manually
3. Delete knowledge
4. View knowledge base
5. Quit

Enter your choice (1-5): 1

Available LLM Providers:
1. Anthropic (Claude-3-Sonnet)
2. OpenAI (O1-Mini)
3. Gemma3:4B
4. DeepSeek-R1:7B

Select LLM provider (1-4, default is 1): 1

What would you like to model? A coffee mug with a hexagonal pattern and a curved handle

=== STEP 1: KEYWORD EXTRACTION ===
Analyzing description...

Keyword Analysis Results:
------------------------------
Core Type: mug
Modifiers: [coffee, hexagonal, pattern, curved, handle]
Compound Type: coffee mug
------------------------------

Do you accept these keywords? (yes/no): yes

=== STEP 2: TECHNICAL ANALYSIS ===
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

=== STEP 3: FINDING SIMILAR EXAMPLES ===
Searching knowledge base...

Found 2 relevant examples (threshold: 0.7):

Example (Score: 0.87):
ID: container_a4f7e9b2
Score Breakdown:
  component_match: 0.92
  step_back_match: 0.88
  geometric_match: 0.85
  feature_match: 0.81

Example (Score: 0.76):
ID: mug_f3e1c8b7
Score Breakdown:
  component_match: 0.79
  step_back_match: 0.74
  geometric_match: 0.82
  feature_match: 0.71

=== STEP 4: GENERATING SCAD CODE ===
Generating OpenSCAD code...
Thinking.....................

OpenSCAD code has been generated and saved to 'output.scad'

Generated Code:
----------------------------------------
// Coffee Mug with Hexagonal Pattern and Curved Handle
// Author: Generated by 3D Modelling with Generative AI

// Parameters
$fn = 100; // Smoothness
mug_height = 95; // Height of mug
mug_radius = 40; // Radius of mug
wall_thickness = 3; // Thickness of the mug wall
handle_thickness = 7; // Thickness of the handle
...
----------------------------------------

Would you like to add this example to the knowledge base? (y/n): y
Example added to knowledge base!
```

## Project Structure

- `rag_3d_modeler.py`: Main application interface
- `OpenSCAD_Generator.py`: Core generation logic
- `enhanced_scad_knowledge_base.py`: Vector-based knowledge management
- `LLM.py`: LLM provider management
- `KeywordExtractor.py`: Extracts core concepts from descriptions
- `metadata_extractor.py`: Analyzes models for metadata
- `conversation_logger.py`: Tracks interactions for future improvement
- `prompts.py`: System prompts and templates
- `constant.py`: System constants and configurations

## Example Output Renders

![Example 1](https://i.imgur.com/example1.png)
![Example 2](https://i.imgur.com/example2.png)

## Requirements

- Python 3.8+
- OpenSCAD installed for rendering models
- API keys for Anthropic and/or OpenAI (optional)
- Ollama installed (for example validation and local models)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

### Areas for Future Development

- Integration with 3D printing services
- Advanced visualization capabilities
- Expanded material property support
- UI/UX improvements
- Enhanced version control for models

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenSCAD community
- Anthropic and OpenAI for their LLM APIs
- Ollama project for local model support
- The ChromaDB team for their vector database implementation