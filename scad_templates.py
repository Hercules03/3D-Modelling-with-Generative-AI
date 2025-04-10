# Define SCAD templates for different object types
SCAD_TEMPLATES = {
    "container": """
        module {object_type}({params}) {{
            difference() {{
                {outer_shape}
                {inner_space}
            }}
        }}
    """,
    
    "mechanical": """
        module {object_type}({params}) {{
            union() {{
                {base_shape}
                {mechanical_features}
            }}
        }}
    """,
    
    "furniture": """
        module {object_type}({params}) {{
            // Main body
            union() {{
                {base_structure}
                {additional_parts}
            }}
        }}
    """,
    
    "decorative": """
        module {object_type}({params}) {{
            // Decorative object with artistic features
            union() {{
                {main_shape}
                // Decorative elements
                {decorative_features}
            }}
        }}
    """,
    
    "tableware": """
        module {object_type}({params}) {{
            // Tableware object for food/drinks
            difference() {{
                {main_body}
                {cavity}
            }}
            // Additional features
            {additional_features}
        }}
    """,
    
    "architectural": """
        module {object_type}({params}) {{
            // Architectural element with structural features
            union() {{
                {main_structure}
                {details}
            }}
        }}
    """,
    
    "organizer": """
        module {object_type}({params}) {{
            // Storage/organizer with compartments
            union() {{
                {outer_shell}
                // Internal divisions
                {compartments}
            }}
        }}
    """,
    
    "electronic": """
        module {object_type}({params}) {{
            // Electronic device with case and features
            union() {{
                {main_case}
                {internal_features}
            }}
            // Cutouts for buttons, ports, etc.
            difference() {{
                {case_with_features}
                {cutouts}
            }}
        }}
    """,
    
    "instrument": """
        module {object_type}({params}) {{
            // Physical tool or instrument
            union() {{
                {handle_section}
                {functional_section}
            }}
        }}
    """,
    
    "jewelry": """
        module {object_type}({params}) {{
            // Decorative jewelry piece
            union() {{
                {main_structure}
                // Decorative details and gems
                {decorative_details}
            }}
        }}
    """,
    
    "utility": """
        module {object_type}({params}) {{
            // Utility object with functional features
            difference() {{
                union() {{
                    {main_body}
                    {functional_features}
                }}
                {cutouts_and_holes}
            }}
        }}
    """,
    
    "toy": """
        module {object_type}({params}) {{
            // Toy or game piece
            union() {{
                {main_body}
                {play_features}
            }}
        }}
    """,
    
    "enclosure": """
        module {object_type}({params}) {{
            // Enclosure with lid
            difference() {{
                {outer_box}
                {inner_cavity}
            }}
            
            // Separate lid component
            module {object_type}_lid() {{
                {lid_design}
            }}
        }}
    """,
    
    "fixture": """
        module {object_type}({params}) {{
            // Fixture for mounting/holding
            difference() {{
                {main_body}
                // Mounting holes and features
                {mounting_features}
            }}
        }}
    """,
    
    "modular": """
        // Modular design with interchangeable parts
        module {object_type}_base({params}) {{
            {base_structure}
        }}
        
        module {object_type}_connector({params}) {{
            {connector_design}
        }}
        
        module {object_type}_attachment({params}) {{
            {attachment_design}
        }}
        
        // Complete assembly
        module {object_type}({params}) {{
            {assembly_code}
        }}
    """
}

# Template selection helper function
def select_template_for_object(object_type, modifiers=None, step_back_analysis=None):
    """
    Select the most appropriate template based on object type and modifiers.
    
    Args:
        object_type (str): The core type of the object
        modifiers (list): Optional list of modifiers/descriptors
        step_back_analysis (dict): Optional step-back analysis results
        
    Returns:
        str: The template name to use
    """
    # Convert inputs to lowercase for easier matching
    if object_type:
        object_type = object_type.lower()
    if modifiers:
        modifiers = [m.lower() for m in modifiers]
    else:
        modifiers = []
    
    # Container-like objects
    container_types = ['box', 'cup', 'bowl', 'vase', 'pot', 'jar', 'bottle', 'container', 'bin', 'basket', 'tray']
    container_modifiers = ['hollow', 'storage', 'container', 'holder', 'dish']
    
    # Furniture
    furniture_types = ['chair', 'table', 'desk', 'shelf', 'bookcase', 'bed', 'couch', 'sofa', 'cabinet', 'dresser', 'stool']
    
    # Mechanical objects
    mechanical_types = ['gear', 'wheel', 'lever', 'hinge', 'joint', 'spring', 'mount', 'bracket', 'adapter', 'mechanism']
    mechanical_modifiers = ['mechanical', 'moving', 'kinetic', 'articulated', 'adjustable']
    
    # Decorative objects
    decorative_types = ['statue', 'figurine', 'ornament', 'sculpture', 'decoration', 'model', 'art']
    decorative_modifiers = ['decorative', 'artistic', 'display', 'sculptural']
    
    # Tableware
    tableware_types = ['plate', 'cup', 'mug', 'glass', 'fork', 'spoon', 'knife', 'bowl', 'dish', 'saucer']
    
    # Architectural elements
    architectural_types = ['column', 'arch', 'dome', 'wall', 'roof', 'floor', 'balcony', 'stair', 'door', 'window']
    
    # Organizers
    organizer_types = ['organizer', 'drawer', 'divider', 'separator', 'holder', 'stand', 'rack', 'hanger']
    organizer_modifiers = ['storage', 'organization', 'sorted', 'compartment', 'divided']
    
    # Electronic devices
    electronic_types = ['phone', 'case', 'computer', 'keyboard', 'mouse', 'controller', 'charger', 'device']
    electronic_modifiers = ['electronic', 'digital', 'charging', 'power', 'smart']
    
    # Instruments/tools
    instrument_types = ['tool', 'handle', 'driver', 'wrench', 'hammer', 'saw', 'opener', 'instrument']
    
    # Jewelry
    jewelry_types = ['ring', 'necklace', 'bracelet', 'pendant', 'chain', 'earring', 'brooch', 'jewel']
    jewelry_modifiers = ['ornamental', 'precious', 'gem', 'decorative']
    
    # Utility objects
    utility_types = ['hook', 'clip', 'holder', 'stand', 'mount', 'bracket', 'support', 'spacer', 'fastener']
    utility_modifiers = ['utility', 'functional', 'practical', 'useful']
    
    # Toys and game pieces
    toy_types = ['toy', 'game', 'figure', 'piece', 'block', 'ball', 'doll', 'puzzle']
    toy_modifiers = ['playful', 'interactive', 'educational', 'recreational']
    
    # Enclosures with lids
    enclosure_types = ['box', 'case', 'enclosure', 'container', 'chest', 'caddy']
    enclosure_modifiers = ['lid', 'cover', 'top', 'closing', 'hinged']
    
    # Fixtures and mounting components
    fixture_types = ['fixture', 'mount', 'bracket', 'clamp', 'stand', 'holder', 'grip']
    fixture_modifiers = ['mounting', 'fixed', 'holding', 'securing', 'attaching']
    
    # Modular designs
    modular_types = ['system', 'module', 'component', 'set', 'kit']
    modular_modifiers = ['modular', 'attachable', 'interlocking', 'interchangeable', 'connectable']
    
    # Default to utility if no specific match is found
    template_name = "utility"
    
    # Check for specific type matches
    if object_type in container_types or any(mod in container_modifiers for mod in modifiers):
        template_name = "container"
    elif object_type in furniture_types:
        template_name = "furniture"
    elif object_type in mechanical_types or any(mod in mechanical_modifiers for mod in modifiers):
        template_name = "mechanical"
    elif object_type in decorative_types or any(mod in decorative_modifiers for mod in modifiers):
        template_name = "decorative"
    elif object_type in tableware_types:
        template_name = "tableware"
    elif object_type in architectural_types:
        template_name = "architectural"
    elif object_type in organizer_types or any(mod in organizer_modifiers for mod in modifiers):
        template_name = "organizer"
    elif object_type in electronic_types or any(mod in electronic_modifiers for mod in modifiers):
        template_name = "electronic"
    elif object_type in instrument_types:
        template_name = "instrument"
    elif object_type in jewelry_types or any(mod in jewelry_modifiers for mod in modifiers):
        template_name = "jewelry"
    elif object_type in utility_types or any(mod in utility_modifiers for mod in modifiers):
        template_name = "utility"
    elif object_type in toy_types or any(mod in toy_modifiers for mod in modifiers):
        template_name = "toy"
    elif object_type in enclosure_types and any(mod in enclosure_modifiers for mod in modifiers):
        template_name = "enclosure"
    elif object_type in fixture_types or any(mod in fixture_modifiers for mod in modifiers):
        template_name = "fixture"
    elif object_type in modular_types or any(mod in modular_modifiers for mod in modifiers):
        template_name = "modular"
    
    # If step-back analysis is provided, use it for additional judgment
    if step_back_analysis and 'principles' in step_back_analysis:
        principles = [p.lower() for p in step_back_analysis['principles']]
        
        # Check for container principles
        if any(term in ' '.join(principles) for term in ['hollow', 'container', 'inside', 'outside', 'empty space', 'interior']):
            template_name = "container"
        
        # Check for mechanical principles
        if any(term in ' '.join(principles) for term in ['movement', 'mechanical', 'moving parts', 'rotate', 'joint', 'motion']):
            template_name = "mechanical"
    
    return template_name

# Function to generate template parameters based on object type and step-back analysis
def generate_template_params(object_type, modifiers=None, step_back_result=None):
    """
    Generate appropriate parameters for a template based on object characteristics.
    
    Args:
        object_type (str): Core type of the object
        modifiers (list): Optional modifiers/descriptors
        step_back_result (dict): Optional step-back analysis results
        
    Returns:
        dict: Parameters to use with the template
    """
    # Default parameters
    params = {
        "object_type": object_type,
        "params": "height=10, width=20, depth=15, wall_thickness=2",
    }
    
    # Initialize template-specific parameters
    for key in ["outer_shape", "inner_space", "base_shape", "mechanical_features",
                "base_structure", "additional_parts", "main_shape", "decorative_features",
                "main_body", "cavity", "additional_features", "main_structure", "details",
                "outer_shell", "compartments", "main_case", "internal_features",
                "case_with_features", "cutouts", "handle_section", "functional_section",
                "decorative_details", "functional_features", "cutouts_and_holes",
                "play_features", "outer_box", "inner_cavity", "lid_design",
                "mounting_features", "base_structure", "connector_design",
                "attachment_design", "assembly_code"]:
        params[key] = "// Add your code here"
    
    # Apply step-back analysis if available
    if step_back_result and 'abstractions' in step_back_result:
        abstractions = step_back_result['abstractions']
        params["main_shape"] = "// Main shape based on: " + ", ".join(abstractions)
        params["base_shape"] = "// Base shape based on: " + ", ".join(abstractions)
        params["outer_shape"] = "// Outer shape based on: " + ", ".join(abstractions)
    
    # If container type, provide appropriate default inner space
    if "container" in params.get("object_type", "").lower() or "box" in params.get("object_type", "").lower():
        params["outer_shape"] = "cube([width, depth, height], center=true);"
        params["inner_space"] = "translate([0, 0, wall_thickness])\n            cube([width-wall_thickness*2, depth-wall_thickness*2, height], center=true);"
    
    # If furniture, provide default structure
    if params.get("object_type", "").lower() in ["chair", "table", "desk", "shelf"]:
        params["base_structure"] = "// Base structure\ncube([width, depth, height/10], center=true);"
        params["additional_parts"] = "// Legs\nfor (i = [-1, 1], j = [-1, 1]) {\n            translate([i*width/2.2, j*depth/2.2, -height/4])\n            cylinder(h=height/2, r=width/20);\n        }"
    
    return params

# Helper function to apply a template
def apply_template(template_name, params):
    """
    Apply the given template with parameters.
    
    Args:
        template_name (str): Name of the template to use
        params (dict): Parameters to use with the template
        
    Returns:
        str: The resulting SCAD code
    """
    if template_name not in SCAD_TEMPLATES:
        template_name = "utility"  # Use utility as fallback
    
    template = SCAD_TEMPLATES[template_name]
    
    # Format the template with parameters
    try:
        return template.format(**params)
    except KeyError as e:
        # If a parameter is missing, add default value and try again
        params[str(e).strip("'")] = "// Add your code here"
        return template.format(**params)