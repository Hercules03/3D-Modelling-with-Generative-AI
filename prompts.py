# Basic OpenSCAD knowledge
BASIC_KNOWLEDGE = """
OpenSCAD is a 3D modeling tool that uses a scripting language to create 3D objects. Here are some basic concepts:

Basic shapes (3D):
1. cube(size = [x,y,z], center = true/false); - Creates a rectangular prism
2. sphere(r=radius); - Creates a sphere
3. cylinder(h = height, r1 = BottomRadius, r2 = TopRadius, center = true/false); - Creates a cylinder
4. polyhedron( points = [ [X0, Y0, Z0], [X1, Y1, Z1], ... ], faces = [ [P0, P1, P2, P3, ...], ... ], convexity = N); - Creates complex shapes

Transformations:
1. translate([x,y,z]) - Moves object
2. rotate(a = deg_a, v = [x, y, z]) - Rotates object (degrees)
3. scale([x,y,z]) - Scales object
4. resize([x,y,z],auto,convexity) - Modifies the size of the child object to match the given x,y,z.
5. mirror([x,y,z]) - Creates a mirror image of a child element across a specified plane
6. multmatrix(m) - Multiplies the geometry of all child elements with the given affine transformation matrix
7. offset(r|delta,chamfer) - generates a new 2d interior or exterior outline from an existing outline
8. hull() - Displays the convex hull of child nodes.
9. minkowski(convexity) - Creates a minkowski sum of child nodes.

Boolean operations:
1. union() - Combines objects
2. difference() - Subtracts second object from first
3. intersection() - Shows overlap between objects

Basic Syntax:
1. Variables: var = value; (e.g., radius = 10;)
2. Comments: // Single line comment, /* Multi-line comment */
3. Mathematical Operators: +, -, *, /, %, ^
4. Comparison Operators: ==, !=, <, <=, >, >=
5. Logical Operators: &&, ||, !
6. Conditional Statement: if (condition) { ... } else { ... }
7. Loops:
- for (i = [start:step:end]) { ... }
- for (i = [value1, value2, ...]) { ... }
8. Modules:
- Define: module name(parameters) { ... }
- Call: name(parameters);
9. Functions:
- Define: function name(parameters) = expression;
- Call: name(parameters)
10. Echo Command: echo("text", variable);
11. Children: Using $children and children() for module composition

Templates:
The system provides templates for common object types:
- container: For hollow objects with inner space
- mechanical: For objects with moving or functional parts
- furniture: For chairs, tables, and other furniture pieces
- decorative: For artistic and decorative objects
- tableware: For plates, cups, and other food-related items
- architectural: For building and structural elements
- organizer: For storage and organizing objects
- electronic: For device cases and holders
- instrument: For tools and functional instruments
- jewelry: For decorative wearable items
- utility: For practical everyday objects
- toy: For playful and recreational items
- enclosure: For cases with lids or covers
- fixture: For mounting and attachment components 
- modular: For systems with interchangeable parts

You can use these templates by calling the appropriate module template.
"""

# System prompt for Ollama models
OLLAMA_SYSTEM_PROMPT = """You are an expert at 3D modeling and OpenSCAD programming. Your task is to help generate OpenSCAD code for creating 3D models based on user descriptions.

Focus on:
1. Geometric principles and mathematical relationships
2. Clean, efficient, and well-structured code
3. Clear comments explaining the design
4. Proper use of OpenSCAD operations and transformations

Format your responses with clear sections for thinking process and code."""

# Keyword extractor system prompt
KEYWORD_EXTRACTOR_SYSTEM_PROMPT = """You are a keyword extraction expert. Your task is to analyze input descriptions and extract their key components.

For each new input description you receive:
1. Identify the main object or concept (core_type)
2. Extract any descriptive words or modifiers
3. Combine them into a meaningful compound type

NEVER return example responses - always analyze the actual input.

You must return a valid JSON object with exactly this structure:
{
    "core_type": "main object type",
    "modifiers": ["list of important modifiers"],
    "compound_type": "full compound type if applicable"
}"""

# Keyword extractor prompt template
KEYWORD_EXTRACTOR_PROMPT = """**Role:** You are an expert system specialized in extracting object types and their modifiers from short text descriptions.

**Task:** Analyze the input `Description` provided below. Identify the primary object being described, any words or short phrases modifying it, and construct a combined term.

**Input Description:**
<<query>>

**Instructions:**

1.  **Focus Solely on Input:** Analyze *only* the text provided in the `Input Description` section above. Ignore all other text in this prompt, including examples.
2.  **Identify Core Object:** Determine the single noun that represents the main object or concept being described. This is the `core_type`.
3.  **Identify Modifiers:** Find all words or short phrases that directly describe or specify the `core_type`. These are the `modifiers`. Maintain their logical order as found in the description or as they would naturally precede the core type.
4.  **Construct Compound Type:** Create a string that combines the `modifiers` (if any) and the `core_type` in a natural language order. If there are no modifiers, the `compound_type` is identical to the `core_type`.
5.  **Output Format:** Return *only* a single JSON object containing exactly the following three keys:
    *   `core_type`: (String) The single noun identified as the main object (e.g., "table", "wheel", "rim").
    *   `modifiers`: (Array of Strings) An array containing the identified modifying words/phrases in order. If no modifiers are found, return an empty array `[]`. (e.g., [], ["car"], ["circular top"], ["car", "5 strokes"]).
    *   `compound_type`: (String) The combined natural language representation (e.g., "table", "car wheel", "circular top table", "car rim with 5 strokes").

**Constraint Checklist (Internal):**
*   Did I analyze ONLY the `<<query>>` text? YES / NO
*   Is the `core_type` a single noun representing the main object? YES / NO
*   Are `modifiers` in an array of strings, in logical order? YES / NO (or empty array `[]`)
*   Does `compound_type` combine modifiers and core type naturally? YES / NO
*   Is the output ONLY the specified JSON object? YES / NO

**Examples (For clarification ONLY - DO NOT process these):**

*   Input: "I want a table" -> Output: `{"core_type": "table", "modifiers": [], "compound_type": "table"}`
*   Input: "Create a car wheel" -> Output: `{"core_type": "wheel", "modifiers": ["car"], "compound_type": "car wheel"}`
*   Input: "Give me a table with a circular top please" -> Output: `{"core_type": "table", "modifiers": ["circular top"], "compound_type": "circular top table"}`
*   Input: "Create a rim of a car with 5 strokes" -> Output: `{"core_type": "rim", "modifiers": ["car", "5 strokes"], "compound_type": "car rim with 5 strokes"}`

**Final Strict Instruction:** Your entire response must be *only* the JSON object derived from the `Input Description: <<query>>`. Do not include any greetings, explanations, or other text before or after the JSON."""

# Step-back prompt template
STEP_BACK_PROMPT_TEMPLATE = """When creating a 3D model for "{Object}"(a {Type}-type device with {Modifiers} modifiers), what are the fundamental principles and high-level concepts I should consider before implementation?

{description}

Please analyze this object and provide me with a structured technical analysis with EXACTLY these three sections:

CORE PRINCIPLES:
- [First core principle]
- [Second core principle]
- [Additional core principles]

SHAPE COMPONENTS:
- [First shape component]
- [Second shape component]
- [Additional shape components]

IMPLEMENTATION STEPS:
1. [First implementation step]
2. [Second implementation step]
3. [Third implementation step]
4. [Additional implementation steps]

Important instructions:
1. Use bullet points for CORE PRINCIPLES and SHAPE COMPONENTS sections, and numbered steps for IMPLEMENTATION STEPS section.
2. Do not include any other sections or explanatory text.
3. RESPOND IN ENGLISH ONLY. Do not use any other language.
"""

# Main OpenSCAD generator prompt template
OPENSCAD_GNERATOR_PROMPT_TEMPLATE = """You are an expert in OpenSCAD 3D modeling. Your task is to generate OpenSCAD code based on the user's description.

BASIC KNOWLEDGE:
{basic_knowledge}

RELEVANT EXAMPLES:
{examples}

STEP-BACK ANALYSIS:
{step_back_analysis}

TEMPLATE SUGGESTION:
{template_suggestion}

PARAMETER SUGGESTIONS:
{parameter_suggestions}

WEB CONTENT AND REFERENCE INFORMATION:
{web_content}

USER REQUEST:
{request}

Please generate OpenSCAD code that satisfies the user's request. Follow these guidelines:
1. Use clear variable names and comments
2. Break down complex shapes into modules 
3. Use proper indentation and formatting
4. Include helpful comments explaining the code
5. Make the design parametric where appropriate (using variables for key dimensions)
6. Implement the design following the step-back analysis principles
7. Use techniques from relevant examples when applicable
8. Consider the suggested template structure if it's appropriate for this design

CRITICAL INSTRUCTIONS:
- Your response must ONLY contain the OpenSCAD code and NOTHING else
- Do NOT include any explanations, questions, or other text outside the code block
- Do NOT ask for clarification - implement the best solution based on the information provided
- ALWAYS enclose your code in triple backtick code blocks with the scad tag, like this:

```scad
// Your code here
```

OpenSCAD code:"""

METADATA_EXTRACTION_PROMPT = """You are an expert in 3D modeling and OpenSCAD. Analyze this 3D model description and extract key metadata.

Description: {description}

step-back analysis:
{step_back_analysis}

Then, extract the following metadata and format it as a valid JSON object with these fields:
1. "object_type": Main category/type of the object (e.g., "mug", "chair", "box")
2. "dimensions": Dictionary of any mentioned measurements or proportions
3. "features": List of key characteristics or components
4. "materials": List of any specified or implied materials
5. "complexity": One of ["SIMPLE", "MEDIUM", "COMPLEX"] based on features and structure
6. "style": Design style (e.g., "Modern", "Traditional", "Industrial", "Minimalist")
7. "use_case": Primary intended use or purpose
8. "geometric_properties": List of key geometric characteristics (e.g., "symmetrical", "curved", "angular")
9. "technical_requirements": List of specific technical considerations
10. "step_back_analysis": {{
    "core_principles": ["list of fundamental principles and concepts"],
    "shape_components": ["list of basic geometric shapes and parts"],
    "implementation_steps": ["list of ordered steps for construction"]
}}

Example response format:
{{
    "object_type": "sword",
    "dimensions": {{
        "length": "100cm",
        "blade_width": "5cm"
    }},
    "features": ["blade", "hilt", "guard", "pommel"],
    "materials": ["metal", "leather"],
    "complexity": "MEDIUM",
    "style": "Fantasy",
    "use_case": ["Role-playing", "Display", "Decoration"],
    "geometric_properties": ["symmetrical", "tapered", "angular"],
    "technical_requirements": ["boolean operations", "smooth transitions"],
    "step_back_analysis": {{
        "core_principles": [
            "Blade geometry follows historical sword designs",
            "Guard provides hand protection",
            "Weight distribution affects balance"
        ],
        "shape_components": [
            "Elongated tapered cylinder for blade",
            "Cross-shaped guard",
            "Cylindrical grip",
            "Spherical pommel"
        ],
        "implementation_steps": [
            "Create blade using cylinder and scale",
            "Add cross-guard using cube and transforms",
            "Form grip with cylinder",
            "Attach spherical pommel",
            "Apply boolean operations for details"
        ]
    }}
}}

Only include fields where information can be confidently extracted from the description.
Format numbers consistently (use metric units when possible).
If a field cannot be determined, omit it from the JSON rather than using placeholder values.

Return ONLY the JSON object, no additional text or explanation."""

CATEGORY_ANALYSIS_PROMPT = """Analyze the following 3D object and categorize it using our standardized categories and properties.

Object Type: {object_type}
Description: {description}

Available Categories:
{categories_info}

Available Properties:
{properties_info}

Instructions:
1. Select the most appropriate categories from the list above (you can select multiple if applicable)
2. For each selected property type, choose the most appropriate value from its predefined options
3. Suggest similar objects from the examples in our standard categories
4. ALWAYS include the object_type field in your response, preserving the input object_type

Respond in JSON format:
{{
    "categories": ["list of categories from standard categories only"],
    "properties": {{
        "property_name": "value from predefined options only"
    }},
    "similar_objects": ["list of similar objects from our standard examples only"],
    "object_type": "{object_type}"
}}

Ensure all categories and property values exactly match the provided standard options.
It is CRITICAL that you include the "object_type" field with the same value as provided in the input."""

WEB_CONTENT_GRADER_PROMPT = """You are a grader assessing relevance of retrieved web content to a user question.
    If the web content contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the web content is relevant to the question."""
    
STEP_BACK_HALLUCINATION_CHECKER_SYSTEM_PROMPT = """You are a grader assessing whether a step back analysis is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the analysis is grounded in / supported by the set of facts.
'No' means the analysis contains hallucinations or information not supported by the facts.

When reviewing the analysis:
1. Verify that specific details, measurements, proportions, or technical aspects mentioned in the analysis are supported by the retrieved content
2. Check that any assertions about the structure, components, or design of the object are based on information in the source material
3. Ensure that any recommendations for modeling techniques are reasonable given the source material
4. Be especially careful about precise numerical values, material properties, or specific design details that would need evidence

Your goal is to identify when the analysis includes speculative or invented information that isn't supported by the source material.
"""
        
# Step-back grader prompt
STEP_BACK_GRADER_PROMPT = """You are an expert evaluator of 3D modeling analyses. Your job is to rate step-back analyses for 3D modeling tasks on a scale of 0-10.

When evaluating a step-back analysis for 3D modeling, consider:

1. Geometric Completeness: Does it identify all core geometric features?
2. Component Breakdown: Does it break the object into logical components?
3. Spatial Relationships: Does it address how components relate to each other?
4. Technical Feasibility: Would the implementation steps work in a 3D modeling context?
5. Detail Level: Is there sufficient detail for implementation?
6. Source Utilization: Does it make good use of provided source materials?

Rating Scale:
0-3: Poor - Missing critical components, incorrect approach, insufficient for modeling
4-6: Fair - Has basic components but lacks details or contains inaccuracies
7-8: Good - Comprehensive with minor gaps or improvements needed
9-10: Excellent - Complete, technically sound, ready for implementation

Provide a numeric rating (0-10) and detailed constructive feedback on strengths and areas for improvement."""

QUERY_ANALYSIS_PROMPT = """You are an expert at analyzing 3D modeling queries to find similar examples in a database of SCAD code.

The database contains pairs of user queries and their corresponding SCAD code implementations. Your goal is to help find the closest matching examples that could be adapted for the current user query.

For each query, determine:

1. The best search strategy (semantic, keyword, or hybrid)
2. An enhanced query that highlights essential 3D object characteristics affecting SCAD code implementation
3. Important attributes relevant for filtering (dimensions, connections, mechanical features, etc.)
4. SCAD coding style preference (e.g., 'Modular', 'Parametric', 'Minimalist', 'Functional', 'Organic')
5. Complexity level of the required SCAD code (SIMPLE, MEDIUM, COMPLEX)
6. Aspects of SCAD code implementation that would be important to match (module structures, algorithms, etc.)

Focus specifically on aspects that would affect how the SCAD code is written, as the goal is to find examples with similar code structure that could be adapted to the current query."""