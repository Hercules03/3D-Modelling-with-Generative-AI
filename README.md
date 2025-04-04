# 3D Modeling with Generative AI and LangGraph

A modern Python system for generating OpenSCAD code from natural language descriptions using LangGraph for orchestration and Large Language Models (LLMs) for generation. This project combines structured workflows with AI to simplify the creation of parametric 3D models.

## Core Architecture

### LangGraph Orchestration

The system uses LangGraph to orchestrate a multi-stage, graph-based workflow with:

- **State-managed processing pipeline** with conditional routing
- **Quality-driven feedback loops** to improve generation quality
- **Fault-tolerance** with proper state preservation
- **Modular processing nodes** for maintainable components

### Processing Pipeline

```
Process Input → Keyword Extraction → Search Query Generation → Web Search 
               → Content Filtering → Step-Back Analysis → Quality Assessment 
               → Query Analysis → Example Retrieval → Code Generation
```

Each stage is implemented as a LangGraph node, with conditional edges for intelligent routing based on quality assessments.

## Key Features

### 1. Intelligent Input Processing

- **Keyword Extraction**: Identifies object types, modifiers, and compound terms
- **Web Research**: Performs multi-faceted searches using Tavily API
- **Content Grading**: Filters web content for relevance using LLM-based assessment

### 2. Step-Back Technical Analysis

- **Hierarchical Decomposition**: Breaks complex objects into basic components
- **Structural Insights**: Identifies core principles and implementation approaches
- **Quality Assessment**: Rates analysis quality with detailed feedback
- **Revision Loops**: Automatically improves low-quality analyses

### 3. Smart Example Retrieval

- **Query Analysis**: Tailors search strategy based on object characteristics
- **OpenSCAD Technique Detection**: Identifies required operations (union, difference, extrusion, etc.)
- **Technique-Weighted Ranking**: Prioritizes examples with relevant operations
- **Metadata Enhancement**: Extracts code structures, parameters, and implementation patterns

### 4. Advanced Code Generation

- **Multi-Source Context**: Combines step-back analysis, similar examples, and web research
- **Parameter Extraction**: Identifies and documents key parameters
- **Code Structure Analysis**: Extracts module organization and techniques
- **Enhanced Output**: Generates fully-commented, well-structured OpenSCAD code

## Technical Implementation

### State Management

The system uses LangGraph's `State` class to manage the flow of information:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    input_text: Optional[str]
    extracted_keywords: Optional[Dict[str, Any]]
    search_queries: Optional[List[str]]
    search_results: Optional[Dict[str, List[Dict[str, str]]]]
    filtered_search_results: Optional[Dict[str, List[Dict[str, str]]]]
    step_back_analysis: Optional[Dict[str, Any]]
    analysis_grade: Optional[Dict[str, Any]]
    query_analysis: Optional[Dict[str, Any]]
    similar_examples: Optional[List[Dict[str, Any]]]
    retrieved_metadata: Optional[Dict[str, Any]]
    generated_code: Optional[Dict[str, Any]]
    # ...other state fields
```

### Graph Structure

The workflow is defined as a directed graph with conditional routing:

```python
workflow = StateGraph(State)

# Add nodes
workflow.add_node("process_input", process_input)
workflow.add_node("extract_keywords", extract_keywords)
# ... other nodes

# Add edges
workflow.add_edge(START, "process_input")
workflow.add_edge("process_input", "extract_keywords")
# ... other edges

# Add conditional routing
workflow.add_conditional_edges(
    "grade_step_back_analysis",
    step_back_quality_router,
    {
        "analyze_query": "analyze_query",
        "run_step_back_analysis": "run_step_back_analysis"
    }
)
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenSCAD installed for rendering
- Ollama (optional, for local models)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/3D-Modelling-with-Generative-AI.git
   cd 3D-Modelling-with-Generative-AI
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   Create a `.env` file with:
   ```
   ANTHROPIC_API_KEY=your_anthropic_key
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   ```

5. **Install Ollama models** (optional for local execution):
   ```bash
   ollama pull llama3.2:3b-instruct-q4_K_M
   ollama pull keyword-extractor:latest
   ```

### Usage

Run the main application:
```bash
python 3D_Modelling.py
```

#### Workflow Modes

1. **Generate Mode**: Create new 3D models from text descriptions
2. **Knowledge Input Mode**: Add examples to the knowledge base manually
3. **Knowledge Management**: View and delete examples
4. **Exit**: Quit the application

## Example Session

```
Welcome to the 3D Model Generator!

Available LLM Providers:
1. Anthropic (claude-3-sonnet-20240229)
2. OpenAI (gpt-4-turbo)
3. Gemma (gemma3:4b-it-q8_0)
4. DeepSeek (deepseek-r1:7b)

Select LLM provider (1-4, default is 1): 1

Select an option:
1. Generate a 3D object
2. Input knowledge manually
3. Delete knowledge
4. View knowledge base
5. Quit

Enter your choice (1-5): 1

Enter a description of the object you want to generate: A gear with 12 teeth and a hexagonal center hole

=== PERFORMING KEYWORD EXTRACTION ===

Keyword Analysis Results:
Core Type: gear
Modifiers: [12 teeth, hexagonal, center hole]
Compound Type: gear with 12 teeth and a hexagonal center hole

Do you approve these keywords? (yes/no): yes

=== CREATING SEARCH QUERIES ===

Generated Search Queries:
- gear 12 teeth hexagonal center hole 3D model
- gear 12 teeth hexagonal center hole 3D printing constraints
- gear 12 teeth hexagonal center hole technical requirements
- gear 12 teeth hexagonal center hole 3D model design considerations
- gear 12 teeth hexagonal center hole 3D model polygon count formats

=== PERFORMING SEARCH ===

=== GRADING WEB CONTENT ===

=== PERFORMING STEP-BACK ANALYSIS ===

Core Principles:
- Involute gear tooth profile for smooth power transmission
- Hexagonal center for secure shaft mounting
- 12-tooth configuration for appropriate gear ratio
- Even spacing of teeth for balanced rotation
- Sufficient tooth depth for proper mesh engagement

Shape Components:
- Circular base with appropriate outer diameter
- 12 evenly spaced teeth with involute profile
- Hexagonal center hole
- Optional chamfer or fillet on edges

Implementation Steps:
1. Define key parameters (modules, tooth height, center hole size)
2. Create the base gear outline with appropriate diameter
3. Generate the involute tooth profile
4. Replicate the tooth pattern 12 times around the center
5. Create the hexagonal center hole
6. Apply any final modifications (fillets, chamfers)

=== GRADING STEP-BACK ANALYSIS ===

Analysis quality rating: 9/10 (Good)

=== PERFORMING QUERY ANALYSIS FOR SCAD CODE RETRIEVAL ===

SCAD Code Retrieval Analysis:
Search Strategy: hybrid
Enhanced Query: Parametric gear with 12 teeth and a hexagonal center hole
Important Attributes: ['cylindrical container', 'curved handle', 'hexagonal pattern', 'hollow interior']
Style Preference: Parametric
Complexity: MEDIUM
Code Similarities: ['module-based design', 'pattern generation', 'curved handle construction']
Likely Techniques: ['difference', 'union', 'rotate', 'translate']

=== RETRIEVING SIMILAR SCAD CODE EXAMPLES ===

Found 3 similar SCAD code examples:
Example 1 (Score: 0.842):
ID: gear_001
Techniques Used: difference, cylinder, translate, rotate
Code Metrics: 120 lines, 5 modules, 12 parameters

=== GENERATING SCAD CODE ===

Generated Code Preview:
// --- Beginning of code ---
// Gear with 12 teeth and a hexagonal center hole
// Author: AI Generator
// Date: 2023-07-25

// Parameters
$fn = 120;  // High resolution for smooth curves
num_teeth = 12;  // Number of teeth
pressure_angle = 20;  // Standard pressure angle (degrees)

// Main Parameters
$fn = 100; // Resolution

// --- Middle section ---
module hexagon(size) {
    polygon([
        for (i = [0:5]) 
            [size * cos(i * 60), size * sin(i * 60)]
    ]);
}

// --- End of code ---
module main() {
    difference() {
        mug_body();
        translate([0, 0, -0.1]) cylinder(h=mug_height+0.2, r=mug_inner_radius);
    }
    mug_handle();
}

main();

// Total: 145 lines of OpenSCAD code

Would you like to add this example to the knowledge base? (y/n): y
Example added to knowledge base!
```

## System Components

- **`3D_Modelling.py`**: Main application entry point
- **`generator_graph.py`**: Main LangGraph workflow definition
- **`manual_input_graph.py`**: Graph for manual knowledge input
- **`graph_state_tools.py`**: Processing nodes and state handling
- **`OpenSCAD_Generator.py`**: Core OpenSCAD generation logic
- **`scad_knowledge_base.py`**: Vector database for examples
- **`step_back_analyzer.py`**: 3D object technical analysis
- **`metadata_extractor.py`**: Code and metadata processing
- **`KeywordExtractor.py`**: Input text parsing and extraction
- **`LLM.py`**: LLM provider management
- **`conversation_logger.py`**: Interaction logging
- **`prompts.py`**: System prompts for LLM interactions

## Architecture Diagram

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  User Interface  │────▶│  LangGraph Flow  │────▶│  LLM Processing  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  State Manager   │◀───▶│  Node Processors │◀───▶│     External     │
└──────────────────┘     └──────────────────┘     │       APIs       │
         │                        │               └──────────────────┘
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Vector Search   │◀───▶│  Code Generator  │────▶│  OpenSCAD File   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Acknowledgments

- Langchain and LangGraph teams for the workflow tools
- Anthropic, OpenAI, and other LLM providers
- OpenSCAD community for the modeling language
- Tavily for search API integration
- Chroma for vector database functionality
- Ollama for local model support

## License

MIT License - See LICENSE file for details.