# 3D Modelling with Generative AI
# Final Year Project Report

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Literature Review](#literature-review)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
   - [Core Components](#core-components)
   - [LLM Integration](#llm-integration)
   - [RAG System](#rag-system)
   - [LangGraph Workflows](#langgraph-workflows)
   - [User Interface](#user-interface)
   - [Supporting Modules](#supporting-modules)
6. [Evaluation](#evaluation)
7. [Discussion and Future Work](#discussion-and-future-work)
8. [Conclusion](#conclusion)
9. [References](#references)

## Executive Summary

This project implements an AI-assisted 3D modeling system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to convert natural language descriptions into OpenSCAD code. The system enables users to describe 3D objects in natural language, and through a series of AI-powered analysis steps, generates precise OpenSCAD code that can be exported to standard 3D file formats such as STL, AMF, and others.

The application addresses the gap between conceptualization and implementation in 3D modeling by allowing users without deep technical knowledge of OpenSCAD to create models through simple descriptions. It features a modular architecture with specialized components for knowledge extraction, step-back analysis, code generation, and parameter tuning, all orchestrated through LangGraph workflows.

## Introduction

3D modeling traditionally requires significant expertise in Computer-Aided Design (CAD) software or programming languages like OpenSCAD. This project aims to democratize 3D modeling by leveraging recent advances in AI, particularly Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), to bridge the gap between natural language descriptions and executable 3D modeling code.

### Project Objectives
1. Develop a system that converts natural language descriptions into OpenSCAD code
2. Implement a knowledge base of 3D modeling examples for context-aware generation
3. Create a workflow that breaks down the generation process into logical steps
4. Provide mechanisms for parameter tuning and model refinement
5. Enable export to standard 3D file formats for practical use

### Scope
The system focuses specifically on OpenSCAD as the target modeling language due to its programmatic nature and suitability for parametric design. The project implements a command-line interface for interaction, a vector-based knowledge retrieval system, and integration with multiple LLM providers (Anthropic, OpenAI, and local Ollama models).

## Literature Review

*[Note: This section would typically include academic references to related work in natural language processing, 3D modeling, and LLM applications. You would add relevant papers, techniques, and previous approaches that informed your project.]*

## System Architecture

The application follows a modular architecture organized around LangGraph workflows, which coordinate the operation of specialized components.

### High-Level Architecture

The system consists of five main subsystems:
1. **User Interface** (`3D_Modelling.py`): Command-line interface for user interaction
2. **LangGraph Workflows** (`generator_graph.py`, `manual_input_graph.py`): State machines that orchestrate the generation and knowledge input processes
3. **Knowledge Base** (`scad_knowledge_base.py`): Vector database storing OpenSCAD examples and metadata
4. **LLM Integration** (`LLM.py`): Abstraction for various LLM providers and models
5. **Utility Modules**: Specialized components for tasks like keyword extraction, code validation, and model export

### Data Flow
1. User enters a natural language description of a 3D object
2. The system extracts keywords and performs a step-back analysis of the request
3. Relevant examples are retrieved from the knowledge base
4. The LLM generates OpenSCAD code using the analysis and examples as context
5. The generated code is validated and optionally refined with parameter tuning
6. The final code is exported to the desired 3D file format

### State Management
A central `State` object maintains progressive information throughout the workflow, including:
- Original input text
- Extracted keywords
- Web search results
- Step-back analysis
- Query analysis results
- Retrieved examples
- Generated code
- Export results

## Implementation Details

### Core Components

#### 3D_Modelling.py
The main entry point and command-line interface for the application. This module:
- Initializes the system components (LLM providers, knowledge base, extractors, etc.)
- Presents a menu of options (generate model, input knowledge, tune parameters, etc.)
- Routes user requests to the appropriate workflow
- Handles errors and user interaction

The implementation uses:
```python
def main():
    """Main function for the 3D Model Generator"""
    config = ModelGeneratorConfig(GenerationSettings())
    # Clear and initialize logging
    # ...
    # Ask for LLM model selection
    # ...
    # Initialize session integration
    session_integration = SessionIntegration()
    session_id, current_session = session_integration.start_session()
    # ...
    # Main menu loop
    while True:
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        # Other options...
        # Handle user choices
```

#### LLM.py
Provides abstraction over different LLM providers:
- Defines model configurations for different providers (Anthropic, OpenAI, Gemma, DeepSeek)
- Handles API key management
- Includes caching mechanisms to reduce redundant LLM calls
- Implements error handling and retry logic

Key classes:
- `ModelDefinitions`: Constants for different model identifiers
- `ModelGeneratorConfig`: Configuration for the generator
- `OllamaManager`: Checks and manages Ollama models
- `LLMProvider`: Factory for creating LLM instances

```python
class LLMProvider:
    @staticmethod
    def get_llm(provider="anthropic", temperature=0.7, max_retries=3, model=None, purpose=None, cache_seed=None):
        # Initialize appropriate LLM based on provider
        # Handle API keys, retries, caching, etc.
```

### RAG System

#### scad_knowledge_base.py
The knowledge base is implemented using ChromaDB for vector storage:
- Manages a persistent collection of OpenSCAD examples
- Uses sentence transformers to create embeddings of descriptions
- Stores rich metadata alongside examples for advanced retrieval
- Implements sophisticated re-ranking based on metadata matching

Key methods:
- `add_example`: Add a new example with description, code, and metadata
- `get_examples`: Retrieve relevant examples using similarity search and re-ranking
- `extract_techniques_from_code`: Analyze code to identify OpenSCAD techniques
- `_rank_results`: Re-rank vector search results using metadata similarity

```python
def get_examples(self, description: str, step_back_result: Dict = None, keyword_data: Dict = None, 
               similarity_threshold: float = 0.6, return_metadata: bool = False, max_results: int = 5,
               technique_filters: List[str] = None) -> List[Dict]:
    # Embed query
    # Perform vector search
    # Re-rank results based on metadata
    # Apply technique boosting
    # Return filtered results
```

#### metadata_extractor.py
Extracts structured metadata from descriptions and code:
- Uses LLM to analyze descriptions and identify object properties
- Defines standard categories and properties for 3D objects
- Analyzes code to extract metrics and characteristics
- Provides validation and normalization of metadata

```python
def extract_metadata(self, description, code="", step_back_result=None, keyword_data=None):
    # Extract base metadata from description
    # Analyze code if available
    # Add keyword-based metadata
    # Perform category analysis
    # Validate and normalize
```

### LangGraph Workflows

#### generator_graph.py
Implements the primary workflow for generating OpenSCAD code:
- Defines a `StateGraph` with nodes for each processing step
- Creates specialized node functions with LLM integration
- Implements conditional routing based on quality checks
- Orchestrates the flow from natural language to code generation

```python
def _build_graph(self):
    workflow = StateGraph(State)
    # Add nodes (process_input, extract_keywords, etc.)
    # Add edges between nodes
    # Add conditional edges for quality control
    return workflow.compile(checkpointer=self.memory)
```

#### graph_state_tools.py
Implements the individual node functions used in the LangGraph workflows:
- `process_input`: Preprocesses user input
- `extract_keywords`: Uses LLM to extract key terms
- `create_step_back_analyzer`: Creates an LLM function for step-back analysis
- `analyze_query`: Prepares query for knowledge base retrieval
- `create_scad_code_generator`: Generates OpenSCAD code using LLM

Also defines data models for structured LLM outputs:
- `GradeWebContent`: Binary score for web content relevance
- `GradeStepBackAnalysis`: Rating scale for analysis quality
- `SCADQueryAnalysis`: Analysis of query for finding examples

```python
def create_scad_code_generator(llm, kb_instance):
    def generate_scad_code(state: State) -> Dict:
        # Extract information from state
        # Prepare prompt with context from RAG
        # Generate code using LLM
        # Validate and post-process
        # Return updated state with generated code
```

#### manual_input_graph.py
Implements a workflow for adding examples to the knowledge base:
- Similar structure to generator_graph but for knowledge input
- Reads code from `add.scad` file
- Validates the code using `scad_code_validator.py`
- Extracts metadata and adds to knowledge base

```python
def _store_knowledge(self, state: State) -> dict:
    # Read SCAD code from file
    # Validate the code
    # Prepare metadata from analysis results
    # Add to knowledge base
```

### Supporting Modules

#### KeywordExtractor.py
Extracts structured keyword data from natural language descriptions:
- Uses dedicated LLM instance with specific prompt
- Returns `KeywordData` with core_type, modifiers, and compound_type
- Includes fallback mechanism for LLM failures

```python
def extract_keyword(self, description):
    # Initialize LLM
    # Send prompt
    # Parse JSON response into KeywordData
    # Fallback to simple extraction if needed
```

#### parameter_tuner.py
Provides functionality for tuning parameters in OpenSCAD code:
- Extracts existing parameters from code
- Suggests parameter adjustments based on description
- Allows manual input of parameter changes
- Applies changes to the code string

```python
def suggest_parameter_adjustments(scad_code: str, description: str, llm=None) -> Dict[str, Any]:
    # Extract current parameters
    # Use LLM to suggest adjustments
    # Return structured suggestions
```

#### model_exporter.py
Handles exporting OpenSCAD code to various 3D file formats:
- Checks for OpenSCAD installation
- Creates temporary files
- Executes OpenSCAD CLI commands
- Supports STL, OFF, AMF, 3MF, CSG, DXF, SVG, PNG formats

```python
def export_model(self, scad_code: str, filename: str = "model", 
                export_format: ExportFormat = "stl", resolution: int = 100,
                additional_params: Optional[Dict[str, str]] = None) -> Optional[str]:
    # Save SCAD code to temp file
    # Create OpenSCAD command
    # Run command and capture output
    # Verify file creation
```

#### scad_code_validator.py
Validates OpenSCAD code for syntax and structure:
- Checks for balanced brackets and parentheses
- Identifies missing semicolons
- Verifies basic OpenSCAD elements are present
- Returns detailed validation messages

```python
def validate_scad_code(code: str) -> Tuple[bool, List[str]]:
    # Check code length
    # Check balanced brackets
    # Check for missing semicolons
    # Check for basic OpenSCAD elements
    # Return validation result and messages
```

#### scad_templates.py
Provides template structures for different object types:
- Defines templates for containers, furniture, mechanical parts, etc.
- Includes helper functions to select appropriate templates
- Generates parameters for templates based on object characteristics

```python
def select_template_for_object(object_type, modifiers=None, step_back_analysis=None):
    # Match object type against template categories
    # Consider modifiers and step-back analysis
    # Return the most appropriate template name
```

#### models.py
Defines Pydantic models for data validation:
- `KeywordData`: Extracted keywords structure
- `Parameter`: OpenSCAD parameter structure
- `StepBackAnalysis`: Step-back analysis results
- `QueryAnalysis`: SCAD code retrieval query analysis
- `GenerationResult`: Code generation result with export info

```python
class GenerationResult(BaseModel):
    """Model for the result of code generation"""
    success: bool = Field(description="Whether code generation was successful")
    code: Optional[str] = Field(default=None, description="Generated OpenSCAD code")
    # Other fields...
```

#### prompts.py
Contains the prompt templates for LLM interactions:
- Basic OpenSCAD knowledge injection
- Keyword extraction prompt
- Step-back analysis prompt
- Main OpenSCAD generation prompt
- Metadata extraction prompt
- Various evaluation and analysis prompts

```python
OPENSCAD_GNERATOR_PROMPT_TEMPLATE = """You are an expert in OpenSCAD 3D modeling. Your task is to generate OpenSCAD code based on the user's description.

BASIC KNOWLEDGE:
{basic_knowledge}

RELEVANT EXAMPLES:
{examples}

STEP-BACK ANALYSIS:
{step_back_analysis}

# ...more context...

USER REQUEST:
{request}

# ...instructions...
"""
```

### User Interface

The application uses a command-line interface with the following main functions:
1. **Generate a 3D object**: User enters a description, the system processes it through the generator graph, and returns OpenSCAD code and exports
2. **Input knowledge manually**: User adds code to `add.scad` and provides a description, the system processes it through the manual input graph
3. **Delete knowledge**: Interface for removing examples from the knowledge base
4. **View knowledge base**: Browse and explore stored examples
5. **Manage LLM cache**: Options for clearing or configuring the LLM cache
6. **View generation history**: See previous generations through the session manager
7. **Adjust existing model parameters**: Load a SCAD file and tune its parameters

## Evaluation

*[Note: This section would typically include metrics, user studies, and quantitative/qualitative evaluation of the system. You would add details about how you evaluated the system's performance, accuracy, usability, etc.]*

Potential evaluation metrics for this system include:
- Code generation success rate
- Relevance of retrieved examples (precision/recall)
- LLM response time and resource usage
- Quality of generated OpenSCAD code (complexity, printability)
- User satisfaction with generated models

## Discussion and Future Work

### Strengths
- Modular architecture allows for easy component replacement or improvement
- RAG approach provides grounding in real examples
- Multiple LLM provider support offers flexibility
- Step-back analysis improves understanding before generation
- Parameter tuning enables refinement of generated models

### Limitations
- Command-line interface may limit accessibility for non-technical users
- Depends on LLM understanding of OpenSCAD syntax and concepts
- Knowledge base requires examples to be effective
- May struggle with very complex or novel 3D models

### Future Work
- Web interface for improved accessibility
- Fine-tuning LLMs specifically for OpenSCAD generation
- Integration with more CAD formats beyond OpenSCAD
- Expanding knowledge base with automatic example discovery
- Interactive parameter adjustment interface
- Real-time preview generation during the design process
- Multi-turn conversations for iterative refinement

## Conclusion

The 3D Modelling with Generative AI project demonstrates the potential of combining LLMs with RAG techniques to bridge the gap between natural language and 3D modeling. By breaking down the generation process into distinct steps and providing contextual examples, the system generates usable OpenSCAD code from simple descriptions.

The project shows how domain-specific knowledge (OpenSCAD syntax, 3D modeling principles) can be effectively integrated with general-purpose LLMs through careful prompting, structured workflows, and knowledge retrieval. The resulting system makes 3D modeling more accessible while maintaining the precision required for functional 3D objects.

## References

*[Note: This section would include academic references, libraries used, and other external resources that informed or were used in the project.]*

- LangChain documentation
- LangGraph documentation
- ChromaDB documentation
- OpenSCAD documentation
- Relevant papers on RAG, LLMs, and 3D modeling 