# 3D Modelling with Generative AI

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An intelligent system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to convert natural language descriptions into OpenSCAD code, making 3D modeling accessible to non-technical users.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project bridges the gap between natural language descriptions and 3D modeling by enabling users to describe 3D objects in plain English. Through a series of AI-powered analysis steps, the system generates precise OpenSCAD code that can be exported to standard 3D file formats such as STL, AMF, and others.

Instead of requiring expertise in Computer-Aided Design (CAD) software or programming languages, the system democratizes 3D modeling by converting descriptions like "a coffee mug with a handle" into executable OpenSCAD code that produces a ready-to-print 3D model.

## Features

- **Natural Language to OpenSCAD Code**: Convert plain English descriptions into parametric OpenSCAD code
- **Retrieval-Augmented Generation**: Leverages a knowledge base of examples to improve generation quality
- **Multiple LLM Support**: Compatible with Anthropic (Claude), OpenAI (GPT), and local Ollama models (Gemma, DeepSeek)
- **Step-Back Analysis**: Breaks down complex requests into principles, components, and implementation steps
- **Parameter Tuning**: Intelligent adjustment of parameters for existing models based on new descriptions
- **Template System**: Uses specialized templates for different object types (containers, furniture, mechanical parts)
- **Multiple Export Formats**: Export to STL, AMF, 3MF, CSG, DXF, SVG, and PNG formats
- **Knowledge Base Management**: Add, view, and manage examples in the vector database
- **Validation**: Automatic checks for syntax and structural issues in generated code
- **Code Metrics**: Analysis of code complexity, style, and structure
- **Technique Detection**: Automatic identification of OpenSCAD techniques used in code

## Architecture

The system follows a modular architecture organized around LangGraph workflows:

![Architecture Diagram](https://via.placeholder.com/800x400?text=Architecture+Diagram)

### Main Components:

1. **User Interface** (`3D_Modelling.py`): Command-line interface for user interaction
2. **LangGraph Workflows**: State machines that orchestrate the generation process
   - `generator_graph.py`: Workflow for generating SCAD code from descriptions
   - `manual_input_graph.py`: Workflow for adding examples to the knowledge base
3. **Knowledge Base** (`scad_knowledge_base.py`): Vector database storing OpenSCAD examples and metadata
4. **LLM Integration** (`LLM.py`): Abstraction for various LLM providers
5. **Utility Modules**: Components for keyword extraction, code validation, metadata extraction, and model export
6. **Parameter Tuning** (`parameter_tuner.py`): Intelligent parameter adjustment system
7. **Template System** (`scad_templates.py`): Object-specific code templates

## How It Works

The system follows a sophisticated multi-stage workflow to transform natural language into 3D models:

1. **Keyword Extraction**: Identifies the core object type and modifiers from the user's description
2. **Web Search**: Gathers relevant information about the described object from the web
3. **Step-Back Analysis**: Performs a higher-level analysis to understand:
   - Fundamental principles and physical characteristics
   - Component structure and relationships
   - Implementation approach and techniques
4. **Query Analysis**: Analyzes how to best search the knowledge base for relevant examples
5. **Example Retrieval**: Finds similar examples using vector search with semantic and technique matching
6. **Code Generation**: Synthesizes OpenSCAD code based on:
   - Step-back analysis results
   - Retrieved examples
   - Web search information
   - Appropriate templates based on object type
   - Parameter suggestions
7. **Validation & Refinement**: Checks code for errors and allows parameter tuning
8. **Export**: Converts the OpenSCAD code to various 3D file formats

This approach combines the strengths of LLMs with structured knowledge retrieval and domain-specific techniques to produce high-quality 3D models.

## Installation

### Prerequisites

- Python 3.9+
- OpenSCAD (for exporting models)
- Access to LLM APIs or local Ollama setup

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/3D-Modelling-with-Generative-AI.git
   cd 3D-Modelling-with-Generative-AI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys:
   Create a `.env` file in the project root with your API keys (see `.env.example`):
   ```
   API_KEY=your_anthropic_or_openai_api_key
   ```

5. (Optional) For local models, set up Ollama:
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull gemma:7b
   ollama pull deepseek-coder:6.7b
   ```

## Usage

Run the main application:

```bash
python 3D_Modelling.py
```

### Main Menu Options

1. **Generate a 3D object**: Enter a description to generate OpenSCAD code and 3D models
2. **Input knowledge manually**: Add your own OpenSCAD examples to the knowledge base
3. **Delete knowledge**: Remove examples from the knowledge base
4. **View knowledge base**: Browse and explore stored examples
5. **Manage LLM cache**: Control caching for LLM interactions
6. **View generation history**: Review previous generation sessions
7. **Adjust existing model parameters**: Tune parameters of existing SCAD files
8. **Quit**: Exit the application

## Example

Here's a simple example of using the system:

1. Run the application and select option 1 (Generate a 3D object)
2. Enter a description:
   ```
   Create a cylindrical pen holder with a spiral pattern around the outside
   ```
3. The system will:
   - Extract keywords (core_type: "holder", modifiers: ["pen", "cylindrical", "spiral pattern"])
   - Perform step-back analysis to understand the components and structure
   - Retrieve similar examples from the knowledge base
   - Generate OpenSCAD code
   - Allow parameter tuning
   - Export to STL or other formats if requested

The resulting OpenSCAD code might look something like:

```scad
// Pen holder with spiral pattern
// Parameters
height = 100;
radius = 40;
wall_thickness = 3;
spiral_count = 8;
spiral_width = 5;
spiral_depth = 2;

module pen_holder() {
    difference() {
        // Outer cylinder
        cylinder(h=height, r=radius, $fn=100);
        
        // Inner hollow
        translate([0, 0, wall_thickness])
            cylinder(h=height, r=radius-wall_thickness, $fn=100);
    }
}

module spiral_pattern() {
    for (i = [0:spiral_count-1]) {
        rotate([0, 0, i * (360 / spiral_count)])
        translate([radius, 0, 0])
        rotate([90, 0, 0])
        linear_extrude(height = spiral_width, center = true)
        polygon([
            [0, 0],
            [spiral_depth, height/4],
            [0, height/2],
            [-spiral_depth, 3*height/4],
            [0, height]
        ]);
    }
}

difference() {
    pen_holder();
    spiral_pattern();
}
```

## Project Structure

```
3D-Modelling-with-Generative-AI/
├── 3D_Modelling.py         # Main entry point and CLI
├── LLM.py                  # LLM provider abstractions
├── generator_graph.py      # LangGraph for model generation
├── graph_state_tools.py    # Node functions for LangGraphs
├── manual_input_graph.py   # LangGraph for knowledge input
├── scad_knowledge_base.py  # Vector DB for examples
├── KeywordExtractor.py     # Extract keywords from descriptions
├── parameter_tuner.py      # Tune OpenSCAD parameters
├── scad_templates.py       # Template structures by object type
├── prompts.py              # LLM prompt templates
├── metadata_extractor.py   # Extract metadata from descriptions
├── model_exporter.py       # Export to various formats
├── scad_code_validator.py  # Validate OpenSCAD syntax
├── step_back_analyzer.py   # Higher-level object analysis
├── session_integration.py  # User session management
├── user_session.py         # Session state tracking
├── conversation_logger.py  # Log conversation history
├── models.py               # Pydantic data models
├── constant.py             # Constants and configurations
├── output/                 # Directory for exported models
├── scad_knowledge_base/    # Knowledge base storage
└── requirements.txt        # Python dependencies
```

## Technical Implementation

The system uses several advanced technologies:

- **Vector Search**: ChromaDB for semantic storage and retrieval of examples
- **Embeddings**: SentenceTransformers for high-quality semantic embeddings
- **State Management**: LangGraph for orchestrating complex workflows
- **Content Analysis**: Fuzzy matching and regex-based code analysis
- **Template System**: Object-type detection with specialized code templates
- **Parameter Detection**: Regex-based parameter extraction and analysis
- **Code Validation**: OpenSCAD syntax checking and error correction
- **Structured Output**: Pydantic models for consistent data validation

The knowledge base uses a sophisticated ranking system that considers:
- Semantic similarity between queries and examples
- Technique matching (e.g., union, difference, translate)
- Object type similarity
- Step-back analysis alignment
- Code complexity and structure

## Dependencies

- **LangChain**: Framework for LLM applications
- **LangGraph**: Orchestration for LLM workflows
- **ChromaDB**: Vector database for example storage
- **Pydantic**: Data validation and settings management
- **Sentence Transformers**: Embeddings for vector search
- **OpenSCAD**: Command-line 3D modeling tool
- **fuzzywuzzy**: Fuzzy string matching
- **LLM Providers**: Anthropic, OpenAI, or Ollama

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.