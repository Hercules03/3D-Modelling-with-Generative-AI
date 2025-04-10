# Application Architecture Documentation

## Application Overview

This application is designed to assist users in creating 3D models using the OpenSCAD scripting language. It takes natural language descriptions of objects as input and leverages Large Language Models (LLMs) combined with a Retrieval-Augmented Generation (RAG) system to produce OpenSCAD code. The application features a command-line interface (`3D_Modelling.py`) for user interaction, allowing for model generation, manual knowledge base population, parameter tuning of existing models, and knowledge base management.

## Core Architecture: LangGraph Workflows

The primary logic for processing user requests (both generation and manual knowledge input) is structured using LangGraph state machines.

1.  **State Management:** A central `State` object (defined conceptually in `graph_state_tools.py`, likely as a `TypedDict`) acts as the data carrier throughout the workflow. It accumulates information such as user input, extracted keywords, search results, analysis results, retrieved examples, generated code, etc., as it passes from one node to the next. Pydantic models defined in `models.py` provide structure and validation for complex data within the state (e.g., `KeywordData`, `StepBackAnalysis`, `QueryAnalysis`, `GenerationResult`).
2.  **Graph Definitions:**
    *   `generator_graph.py`: Defines the `Model_Generator_Graph` for the primary workflow of generating SCAD code from a description.
    *   `manual_input_graph.py`: Defines the `Manual_Knowledge_Graph` for analyzing a description and associating it with user-provided SCAD code (from `add.scad`) to add to the knowledge base.
3.  **Nodes (`graph_state_tools.py`):** This file implements the actual functions that perform the work at each step (node) in the graphs. These functions take the current `State` as input, perform their specific task (e.g., keyword extraction, web search, LLM analysis, knowledge base retrieval, code generation), and return updates to the `State`.
4.  **Orchestration (`3D_Modelling.py`):** The main script initializes the necessary components (LLMs, KB, extractors) and invokes the appropriate LangGraph (`.invoke()`) based on user menu choices, passing the initial input and receiving the final state.

## RAG Implementation Details (`scad_knowledge_base.py`)

The RAG system is central to providing relevant context for code generation.

1.  **Storage & Indexing:**
    *   **Vector Store:** Uses ChromaDB (`chromadb.PersistentClient`) configured via `constant.py` (`CHROMA_PERSIST_DIR`) to store and query examples.
    *   **Collection:** A collection named `scad_examples` holds the data.
    *   **Embedding:** Uses `SentenceTransformerEmbeddingFunction` (`all-MiniLM-L6-v2` or potentially `all-mpnet-base-v2` for technical queries identified by `query_is_technical`). When an example is added (`add_example`), its natural language `description` is embedded and stored.
    *   **Metadata:** Crucially, rich metadata is stored alongside the description embedding. This includes:
        *   The actual `code`.
        *   Extracted keywords (`KeywordData`).
        *   Step-back analysis results (`StepBackAnalysis`).
        *   Code metrics (line count, complexity, style) extracted via `analyze_code_structure`.
        *   Identified OpenSCAD techniques used (`extract_techniques_from_code`).
        *   Primary parameters (`extract_parameters_from_code`).
        *   Other metadata extracted by `metadata_extractor.py` (features, geometric properties, style, complexity, etc.).

2.  **Retrieval (`get_examples`):**
    *   **Query Embedding:** The user's input `description` is embedded using the appropriate Sentence Transformer model.
    *   **Vector Search:** ChromaDB performs a similarity search (`collection.query`) against the stored example descriptions.
    *   **Metadata Filtering (Pre/Post-Search):** While the code shows potential for filtering during the query (`filters` argument in `collection.query`), the primary filtering seems to happen *after* the initial vector search, especially for techniques (`technique_filters`).
    *   **Re-Ranking (`_rank_results`):** This is a key step. The initial list of candidates from the vector search is re-ordered based on a weighted similarity score calculated using the *metadata* of the query (derived via `metadata_extractor.py`) and the stored metadata of the example. High weight is given to `object_type_match` and `technique_match`, with lesser weights for features, style, complexity, etc. This refines retrieval beyond just description similarity.
    *   **Technique Boosting:** If the query analysis identifies `techniques_needed`, results containing those techniques receive a score boost, further prioritizing relevant coding patterns.
    *   **Thresholding:** Results below a `similarity_threshold` (currently hardcoded to 0.4 in `get_examples` after boosting) are discarded.

3.  **Augmentation:** The final list of `SimilarExample` objects (including their description, code, metadata, score breakdown, techniques, parameters) is added to the `State` and passed as context to the code generation LLM via the `OPENSCAD_GNERATOR_PROMPT_TEMPLATE`.

## LLM Usage and Specialization for 3D Modeling

LLMs are used throughout the application, specialized for 3D modeling tasks through careful prompting and task delegation:

1.  **LLM Abstraction (`LLM.py`):**
    *   Provides a unified way (`LLMProvider.get_llm`) to instantiate different LLM clients (Anthropic, OpenAI via `ChatOpenAI`; Ollama models like Gemma, DeepSeek via `ChatOllama`).
    *   Handles configuration (API keys via `.env`, base URLs for proxies, temperature).
    *   Defines specific models for specific tasks (`ModelDefinitions`, e.g., `KEYWORD_EXTRACTOR`, `VALIDATOR`).
    *   Integrates caching (`LLMCacheManager`) to reduce redundant calls.

2.  **Prompt Engineering (`prompts.py`):** This is the core of the specialization:
    *   **Code Generation (`OPENSCAD_GNERATOR_PROMPT_TEMPLATE`):** Explicitly instructs the LLM to act as an OpenSCAD expert, provides basic syntax (`BASIC_KNOWLEDGE`), includes the context from the RAG step (retrieved examples, potentially code snippets), step-back analysis, template suggestions (`scad_templates.py`), parameter suggestions (`parameter_tuner.py`), and web context. Crucially, it mandates the output format (SCAD code only, within \`\`\`scad blocks).
    *   **Analysis Prompts (Step-Back, Metadata, Keywords, Category):** These prompts are tailored to extract information relevant to 3D modeling.
        *   `STEP_BACK_PROMPT_TEMPLATE`: Guides the LLM to decompose the request into principles, components, and steps relevant to *constructing* a 3D object.
        *   `METADATA_EXTRACTION_PROMPT`: Asks for specific 3D-related fields (dimensions, features, geometric properties, complexity, style, use_case).
        *   `CATEGORY_ANALYSIS_PROMPT`: Uses predefined 3D modeling categories (`STANDARD_CATEGORIES` / `BASIC_KNOWLEDGE`) for classification.
        *   `QUERY_ANALYSIS_PROMPT`: Focuses the LLM on identifying aspects relevant for finding *similar SCAD code* (strategy, attributes, style, complexity, implementation details/techniques).
    *   **Parameter Tuning Prompts (`parameter_tuner.py`):** Instruct the LLM to analyze existing code and a description to suggest meaningful adjustments to OpenSCAD parameters.

3.  **Structured Output & Validation:**
    *   Many analysis tasks rely on the LLM producing JSON output (e.g., keyword extraction, metadata, category analysis, parameter suggestions). Prompts explicitly request this JSON format.
    *   Pydantic models (`models.py`) are used extensively to parse and validate the structure of data, including LLM outputs where possible (e.g., `KeywordData`), adding robustness.
    *   Grading nodes in the graph (`grade_web_content`, `step_back_analysis_grader`) use LLMs with specific prompts and Pydantic models (`GradeWebContent`, `GradeStepBackAnalysis`) to evaluate intermediate results, enabling conditional logic in the workflow.

4.  **Domain Knowledge:**
    *   Direct injection via `BASIC_KNOWLEDGE` in prompts.
    *   Implicitly via the RAG system retrieving relevant OpenSCAD code examples.
    *   Potentially via `scad_templates.py` providing structural outlines.

## Key Workflows

1.  **Model Generation (`generator_graph.py`)**
    *   User provides description.
    *   Graph runs: Extract keywords -> Analyze description (Step-Back) -> Analyze query for retrieval -> Retrieve relevant examples from KB (RAG) -> Generate SCAD code using description, analysis, and retrieved examples as context.
    *   Result (code, export info) is returned. `ModelExporter` is used (likely within the `generate_scad_code` node or a subsequent step) to create STL/other formats.
2.  **Manual Knowledge Input (`manual_input_graph.py`)**
    *   User adds code to `add.scad` and provides a description via CLI.
    *   Graph runs: Extract keywords -> Analyze description (Step-Back, Query Analysis).
    *   `_store_knowledge` node: Reads `add.scad`, validates code (`scad_code_validator.py`), prepares metadata (using analysis results, extracting techniques/params from code), adds the example (description, code, metadata) to the knowledge base (`scad_knowledge_base.py`).
3.  **Parameter Tuning (`parameter_tuner.py` triggered by `3D_Modelling.py`)**
    *   User provides path to `.scad` file and a description.
    *   `extract_parameters_from_code` gets current parameters.
    *   `suggest_parameter_adjustments` either asks the user for changes or uses the LLM to suggest changes based on the description.
    *   User confirms/selects adjustments.
    *   `apply_parameter_adjustments` modifies the code string.
    *   The updated code is saved and optionally exported (`model_exporter.py`).

## Supporting Modules

*   **`metadata_extractor.py`**: Crucial for generating rich context for RAG ranking and filtering. Uses LLM for analysis.
*   **`KeywordExtractor.py`**: Provides focused keyword data early in the process. Uses LLM.
*   **`parameter_tuner.py`**: Enables refining existing or generated code. Uses LLM.
*   **`scad_code_validator.py`**: Acts as a quality gate for SCAD code (manual input, potentially generated code). Rule-based and regex checks.
*   **`model_exporter.py`**: Handles the final output step, interacting with the external `openscad` CLI.
*   **`scad_templates.py`**: Provides basic structures, likely used as context/suggestions for the generation LLM.
*   **`constant.py`**: Defines shared paths and category/property lists.

## Data Flow

Data flows primarily through the `State` object within the LangGraph workflows. User input initiates the process. LLMs generate analysis (keywords, step-back, metadata, suggestions) and code. The Knowledge Base retrieves relevant examples based on analysis and user input. Validation checks code quality. The Exporter produces the final 3D files. Pydantic models (`models.py`) ensure data consistency at various stages.

This architecture creates a powerful system for generating specialized OpenSCAD code by combining the analytical and generative capabilities of LLMs with the contextual grounding provided by a domain-specific knowledge base accessed via a sophisticated RAG pipeline. 