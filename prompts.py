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
KEYWORD_EXTRACTOR_SYSTEM_PROMPT = """You are a keyword extraction expert. Your task is to extract the main object or concept from descriptions.
Return only the single most relevant keyword, with no additional text or explanation."""

# Keyword extractor prompt template
KEYWORD_EXTRACTOR_PROMPT = """Extract the main object or concept from this description as a single word:
{description}"""

# Step-back prompt template
STEP_BACK_PROMPT_TEMPLATE = """Let's analyze the technical requirements and geometric principles for creating a 3D model based on this request: "{query}"

Please provide a structured technical analysis in this format:

<think>
Consider:
- Required geometric primitives
- Spatial relationships
- Key measurements and proportions
- Technical implementation approach
</think>

<analysis>
CORE PRINCIPLES:
- [List 3-5 key geometric and mathematical concepts]
- [Focus on technical requirements]
- [Include necessary measurements and proportions]

SHAPE COMPONENTS:
- [List primary geometric primitives needed]
- [Describe spatial relationships]
- [Note required transformations]

IMPLEMENTATION STEPS:
1. [Initial setup and base components]
2. [Component creation and positioning]
3. [Assembly and transformations]
4. [Final adjustments and optimization]
</analysis>

Please maintain the XML-style tags and structured format shown above."""

# Main OpenSCAD generator prompt template
OPENSCAD_GNERATOR_PROMPT_TEMPLATE = """You are an expert in OpenSCAD 3D modeling. Your task is to generate OpenSCAD code based on the user's description.

BASIC KNOWLEDGE:
{basic_knowledge}

RELEVANT EXAMPLES:
{examples}

Analysis to consider:
{step_back_analysis}

USER REQUEST:
{request}

Please generate OpenSCAD code that satisfies the user's request. Follow these guidelines:
1. Use clear variable names and comments
2. Break down complex shapes into modules
3. Use proper indentation and formatting
4. Include helpful comments explaining the code
5. Wrap the code in <code> tags or ```scad code blocks

Your response should ONLY contain the OpenSCAD code, properly wrapped in tags. Do not include any explanations or additional text.

OpenSCAD code:"""
        
VALIDATION_PROMPT = """You are a 3D modeling expert. Your task is to determine if this example would be helpful for creating the requested 3D object.

User wants to create: {query}

Retrieved example:
Description: {description}
Object type: {object_type}

Consider:
1. Is this example about the same type of object? (e.g., a cup example for creating a cup)
2. Is this example about a variant of the same object? (e.g., a coffee cup example for creating a cup)
3. Does it have similar structural components that would be useful? (e.g., a mug example for creating a cup)

Answer with ONLY ONE WORD:
- 'useful' - if the example is about the same object type OR a variant OR has similar components
- 'unuseful' - if the example is completely unrelated with no useful components

Answer: """