# Basic OpenSCAD knowledge
BASIC_KNOWLEDGE = """
OpenSCAD is a 3D modeling language that uses programming concepts to create 3D objects.

Basic shapes:
1. cube([width, depth, height]) - Creates a rectangular prism
2. sphere(r=radius) - Creates a sphere
3. cylinder(h=height, r=radius) - Creates a cylinder
4. polyhedron(points=[[x,y,z], ...], faces=[[p1,p2,p3], ...]) - Creates complex shapes

Transformations:
1. translate([x,y,z]) - Moves object
2. rotate([x,y,z]) - Rotates object (degrees)
3. scale([x,y,z]) - Scales object

Operations:
1. union() - Combines objects
2. difference() - Subtracts second object from first
3. intersection() - Shows overlap between objects
"""

OLLAMA_SYSTEM_PROMPT = """
You are an expert in OpenSCAD 3D modeling. Your task is to generate OpenSCAD code based on user descriptions. Only output the code itself, no explanations or JSON. The code should be well-commented and use appropriate measurements.
"""

KEYWORD_EXTRACTOR_SYSTEM_PROMPT = """
"You are a keyword extractor. Extract only the main object name from the user's 3D modeling request. Return ONLY the object name, nothing else."
"""

KEYWORD_EXTRACTOR_PROMPT = """Extract the main object name from this 3D modeling request.
            Only return the object name, no other words or punctuation.
            If there are multiple objects, return the main one.
            If it's a compound word (like 'snowman' or 'birdhouse'), keep it as one word.
            
            Request: {description}
            
            Return ONLY the object name, nothing else."""

OPENSCAD_GNERATOR_PROMPT_TEMPLATE = """You are an expert in 3D modeling with OpenSCAD. Generate OpenSCAD code for the user's request.
        If the request is complex, break it down into simpler shapes and combine them.

        Basic Knowledge:
        {basic_knowledge}

        {examples}

        User Request: {request}

        Important: Generate ONLY pure OpenSCAD code. Do not include any JSON, explanations, or formatting.
        The code should be well-commented and use appropriate measurements.
        """