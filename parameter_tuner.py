import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import copy

logger = logging.getLogger(__name__)

def extract_parameters_from_code(scad_code: str) -> Dict[str, Any]:
    """
    Extract parameters from OpenSCAD code.
    
    Args:
        scad_code: String containing the OpenSCAD code
        
    Returns:
        Dictionary containing parameter names and their values
    """
    # Initialize results
    parameters = {}
    
    # Regular expression to match variable declarations
    # This pattern captures:
    # 1. Variable name
    # 2. Value (number, string, boolean)
    # 3. Optional comment after the declaration
    var_pattern = r'(?:^|\n)\s*([\w_]+)\s*=\s*([^;]+);(?:\s*\/\/\s*(.+?))?(?:\n|$)'
    
    # Find all matches
    matches = re.finditer(var_pattern, scad_code)
    
    for match in matches:
        var_name = match.group(1)
        var_value_str = match.group(2).strip()
        var_comment = match.group(3).strip() if match.group(3) else None
        
        # Try to determine the type and convert the value
        var_value = _parse_parameter_value(var_value_str)
        
        # Store the parameter with type information and comment if available
        parameters[var_name] = {
            "value": var_value,
            "type": _determine_type(var_value),
            "comment": var_comment,
            "original_string": var_value_str
        }
    
    # Look for module parameters
    module_pattern = r'module\s+([\w_]+)\s*\(([^)]*)\)'
    module_matches = re.finditer(module_pattern, scad_code)
    
    for match in module_matches:
        module_name = match.group(1)
        params_str = match.group(2)
        
        # Parse module parameters
        if params_str.strip():
            param_list = params_str.split(',')
            for param in param_list:
                param = param.strip()
                if '=' in param:
                    # Parameter with default value
                    param_name, param_value_str = param.split('=', 1)
                    param_name = param_name.strip()
                    param_value_str = param_value_str.strip()
                    
                    # Parse the value
                    param_value = _parse_parameter_value(param_value_str)
                    
                    # Store in parameters dict if not already present
                    if param_name not in parameters:
                        parameters[param_name] = {
                            "value": param_value,
                            "type": _determine_type(param_value),
                            "comment": f"Parameter for module {module_name}",
                            "original_string": param_value_str,
                            "is_module_param": True,
                            "module_name": module_name
                        }
                else:
                    # Parameter without default value
                    # We still record it but with None value
                    param_name = param.strip()
                    if param_name and param_name not in parameters:
                        parameters[param_name] = {
                            "value": None,
                            "type": "unknown",
                            "comment": f"Parameter for module {module_name} (no default)",
                            "original_string": "",
                            "is_module_param": True,
                            "module_name": module_name
                        }
    
    logger.info(f"Extracted {len(parameters)} parameters from OpenSCAD code")
    return parameters

def _parse_parameter_value(value_str: str) -> Any:
    """
    Parse a parameter value string to determine its actual value and type.
    
    Args:
        value_str: String representation of the parameter value
        
    Returns:
        The parsed value (int, float, bool, str, or list)
    """
    # Remove any trailing comments
    if '//' in value_str:
        value_str = value_str.split('//', 1)[0].strip()
    
    # Boolean values
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
    
    # Try numeric values
    try:
        # Check if it's an integer
        if value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
            return int(value_str)
        
        # Check if it's a float
        return float(value_str)
    except ValueError:
        pass
    
    # Check for lists/vectors
    if value_str.startswith('[') and value_str.endswith(']'):
        # List value
        inner_str = value_str[1:-1].strip()
        if inner_str:
            try:
                items = inner_str.split(',')
                parsed_items = [_parse_parameter_value(item.strip()) for item in items]
                return parsed_items
            except:
                # If parsing as list fails, return as string
                return value_str
        else:
            return []
    
    # Check for strings
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    # Default to returning the string as is
    return value_str

def _determine_type(value: Any) -> str:
    """
    Determine the type of a parameter value.
    
    Args:
        value: The parameter value
        
    Returns:
        String representation of the type
    """
    if value is None:
        return "unknown"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            return "vector"
        else:
            return "list"
    elif isinstance(value, str):
        return "string"
    else:
        return "unknown"

def identify_parameters_from_examples(examples: List[Dict], description: str, llm=None) -> Dict[str, Any]:
    """
    Analyze examples to identify customizable parameters for the new model.
    
    Args:
        examples: List of similar example objects
        description: User's description for the model
        llm: Optional LLM instance for generating suggestions
        
    Returns:
        Dictionary of suggested parameters with default values and descriptions
    """
    print("\nAnalyzing examples to identify customizable parameters...")
    
    # Extract parameters from examples
    all_example_params = []
    for example in examples:
        # Check if example has primary_parameters field
        if 'primary_parameters' in example:
            all_example_params.extend(example.get('primary_parameters', []))
        # Check for code and extract parameters if available
        elif 'code' in example:
            code = example.get('code', '')
            extracted = extract_parameters_from_code(code)
            for name, info in extracted.items():
                # Skip internal variables
                if name.startswith('_') or 'tmp' in name.lower() or 'temp' in name.lower():
                    continue
                    
                all_example_params.append({
                    "name": name,
                    "value": info.get("value", ""),
                    "comment": info.get("comment", "")
                })
    
    # If we don't have any parameters or LLM isn't provided, return empty dict
    if not all_example_params or not llm:
        return {}
    
    # Use LLM to suggest parameters based on examples and description
    print(f"Found {len(all_example_params)} parameters from {len(examples)} examples")
    
    # Create a prompt for the LLM
    prompt = f"""
    I'm creating a 3D model for: {description}
    
    Based on similar examples, I've identified these potential parameters:
    {json.dumps(all_example_params, indent=2)}
    
    Please suggest the most important parameters that should be customizable for this specific model.
    For each parameter, provide:
    1. A name (descriptive and following OpenSCAD conventions)  
    2. A sensible default value
    3. A brief description of what the parameter controls
    
    Create parameters that are relevant to the user's description and would allow good customization.
    Focus only on the most important 5-8 parameters.
    
    Return your response as a JSON object with this structure:
    {{
        "parameters": {{
            "parameter_name": {{
                "value": default_value,
                "description": "Description of parameter", 
                "type": "integer/float/string/boolean/vector"
            }},
            ... additional parameters
        }}
    }}
    """
    
    # Generate suggestions using LLM
    try:
        response = llm.invoke(prompt)
        content = ""
        
        # Handle different response formats (streaming or not)
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        # Extract JSON from response
        # Find JSON content between ``` markers or just extract the JSON object
        json_match = re.search(r'```(?:json)?\s*({\s*"parameters":.+?})\s*```', content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            # Try to find a JSON object without code block markers
            json_match = re.search(r'({[\s\S]*"parameters"[\s\S]*})', content)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Fallback: try to use the entire response
                json_content = content
        
        try:
            # Parse the JSON content
            suggestions = json.loads(json_content)
            return suggestions.get("parameters", {})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response content: {content}")
            return {}
    
    except Exception as e:
        logger.error(f"Error generating parameter suggestions: {str(e)}")
        return {}

def suggest_parameters_from_description(description: str, object_type: str, step_back_result: Dict = None, llm=None) -> Dict[str, Any]:
    """
    Use LLM to suggest appropriate parameters based directly on the user's description.
    
    Args:
        description: User's description of the 3D model to create
        object_type: The core type of the object (e.g., "mug", "chair", "box")
        step_back_result: Optional step-back analysis containing principles and approach
        llm: LLM instance to use for generating suggestions
        
    Returns:
        Dictionary of suggested parameters with default values and descriptions
    """
    if not llm:
        print("No LLM provided for parameter suggestion. Skipping description-based parameter extraction.")
        return {}
    
    # Extract relevant information from step-back analysis if available
    principles = []
    components = []
    approach = []
    
    if step_back_result:
        principles = step_back_result.get('principles', [])
        components = step_back_result.get('abstractions', [])
        approach = step_back_result.get('approach', [])
    
    # Create a prompt for the LLM
    prompt = f"""
    I need to create customizable parameters for a 3D model in OpenSCAD based on this description:
    "{description}"
    
    Core object type: {object_type}
    
    Additional context:
    - Key principles: {", ".join(principles[:3]) if principles else "None provided"}
    - Core components: {", ".join(components[:3]) if components else "None provided"}
    - Implementation approach: {", ".join(approach[:2]) if approach else "None provided"}
    
    Based solely on this description and context, suggest the most important parameters that should be customizable for this specific 3D model.
    
    For each parameter, provide:
    1. A descriptive name following OpenSCAD naming conventions (lowercase with underscores)
    2. An appropriate default value based on typical dimensions/properties for this type of object
    3. A brief description of what the parameter controls
    4. The parameter type (integer, float, boolean, string, or vector)
    
    Focus on creating parameters that would be most useful for customizing this specific object, include:
    - Key dimensions (height, width, depth, etc.)
    - Important structural properties 
    - Style/design elements mentioned in the description
    - Functional aspects from the description
    
    Limit your suggestions to 3-6 of the most essential parameters.
    
    Return your response as a JSON object with this structure:
    {{
        "parameters": {{
            "parameter_name": {{
                "value": default_value,
                "description": "Description of parameter", 
                "type": "integer/float/string/boolean/vector"
            }},
            ... additional parameters
        }}
    }}
    """
    
    try:
        # Get response from LLM
        print("Asking LLM to suggest parameters based on description...")
        response = llm.invoke(prompt)
        
        # Extract the content from an AIMessage object or other response type
        content = ""
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        # Parse JSON response
        # Find JSON block in response
        json_str = None
        
        # First look for JSON block in code tags
        code_tags = [
            ('```json', '```'),
            ('```', '```')
        ]
        
        for start_tag, end_tag in code_tags:
            if start_tag in content and end_tag in content:
                start_idx = content.find(start_tag) + len(start_tag)
                end_idx = content.find(end_tag, start_idx)
                if end_idx > start_idx:
                    json_str = content[start_idx:end_idx].strip()
                    break
        
        # If no code tags, try to find JSON object in the response
        if not json_str:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx].strip()
        
        if json_str:
            try:
                data = json.loads(json_str)
                suggested_params = data.get('parameters', {})
                
                if suggested_params:
                    print(f"LLM suggested {len(suggested_params)} parameters based on description")
                    return suggested_params
                else:
                    print("LLM didn't suggest any parameters from description")
                    # Return default parameters instead of empty dict
                    return {
                        "width": {"value": 100, "description": "Width of the object", "type": "float"},
                        "height": {"value": 150, "description": "Height of the object", "type": "float"},
                        "depth": {"value": 70, "description": "Depth of the object", "type": "float"},
                        "wall_thickness": {"value": 2, "description": "Wall thickness for hollow objects", "type": "float"},
                        "resolution": {"value": 100, "description": "Resolution for curved surfaces ($fn value)", "type": "integer"}
                    }
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response: {str(e)}")
                print(f"Problematic JSON string: {json_str[:100]}...")
                # Return default parameters
                return {
                    "width": {"value": 100, "description": "Width of the object", "type": "float"},
                    "height": {"value": 150, "description": "Height of the object", "type": "float"},
                    "depth": {"value": 70, "description": "Depth of the object", "type": "float"},
                    "wall_thickness": {"value": 2, "description": "Wall thickness for hollow objects", "type": "float"},
                    "resolution": {"value": 100, "description": "Resolution for curved surfaces ($fn value)", "type": "integer"}
                }
        else:
            print("Could not extract JSON from LLM response for parameter suggestions")
            # Return default parameters
            return {
                "width": {"value": 100, "description": "Width of the object", "type": "float"},
                "height": {"value": 150, "description": "Height of the object", "type": "float"},
                "depth": {"value": 70, "description": "Depth of the object", "type": "float"},
                "wall_thickness": {"value": 2, "description": "Wall thickness for hollow objects", "type": "float"},
                "resolution": {"value": 100, "description": "Resolution for curved surfaces ($fn value)", "type": "integer"}
            }
            
    except Exception as e:
        error_message = str(e)
        print(f"Error getting parameter suggestions from LLM: {error_message}")
        import traceback
        traceback.print_exc()
        # Return default parameters instead of empty dict
        return {
            "width": {"value": 100, "description": "Width of the object", "type": "float"},
            "height": {"value": 150, "description": "Height of the object", "type": "float"},
            "depth": {"value": 70, "description": "Depth of the object", "type": "float"},
            "wall_thickness": {"value": 2, "description": "Wall thickness for hollow objects", "type": "float"},
            "resolution": {"value": 100, "description": "Resolution for curved surfaces ($fn value)", "type": "integer"}
        }

def get_user_parameter_input(suggested_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask user for parameter values or use LLM suggestions.
    
    Args:
        suggested_parameters: Dictionary of parameters suggested by LLM
        
    Returns:
        Dictionary of final parameters to use
    """
    if not suggested_parameters:
        print("\nNo customizable parameters were identified.")
        return {}
        
    print("\n=== Customizable Parameters ===")
    print("The following parameters can be customized for your model:")
    
    # Make a copy so we don't modify the original
    final_parameters = copy.deepcopy(suggested_parameters)
    
    # Display parameters
    for i, (param_name, param_info) in enumerate(suggested_parameters.items(), 1):
        value = param_info.get("value")
        description = param_info.get("description", "")
        print(f"{i}. {param_name} = {value} // {description}")
    
    # Ask if user wants to customize
    customize = input("\nWould you like to customize these parameters? (yes/no): ").lower().strip()
    
    if customize in ["yes", "y"]:
        # Get user input for each parameter
        for param_name, param_info in suggested_parameters.items():
            current_value = param_info.get("value")
            description = param_info.get("description", "")
            param_type = param_info.get("type", "")
            
            print(f"\nParameter: {param_name}")
            print(f"Description: {description}")
            print(f"Default value: {current_value} (Type: {param_type})")
            
            new_value = input("Enter new value (or press Enter to keep default): ").strip()
            if new_value:
                # Try to convert value based on type
                try:
                    if param_type == "integer":
                        new_value = int(new_value)
                    elif param_type == "float":
                        new_value = float(new_value)
                    elif param_type == "boolean":
                        new_value = new_value.lower() in ["true", "yes", "y", "1"]
                    elif param_type == "vector":
                        # Parse as vector [x,y,z]
                        vector_parts = new_value.strip('[]').split(',')
                        new_value = [float(p.strip()) for p in vector_parts]
                except ValueError:
                    print(f"Warning: Could not convert '{new_value}' to {param_type}. Using as string.")
                
                final_parameters[param_name]["value"] = new_value
                final_parameters[param_name]["user_provided"] = True
        
        print("\nParameters customized successfully!")
    else:
        print("\nUsing default parameter values.")
    
    return final_parameters

def suggest_parameter_adjustments(scad_code: str, description: str, llm=None) -> Dict[str, Any]:
    """
    Analyze SCAD code and suggest parameter adjustments based on the description.
    
    Args:
        scad_code: The OpenSCAD code to analyze
        description: Description of the desired model
        llm: Optional LLM instance for generating suggestions
        
    Returns:
        Dictionary containing suggested parameter adjustments
    """
    # Extract parameters from code
    parameters = extract_parameters_from_code(scad_code)
    
    # If no parameters found, return empty suggestions
    if not parameters:
        logger.warning("No parameters found in the SCAD code")
        return {
            "success": False,
            "error": "No parameters found in the SCAD code",
            "parameters": {},
            "suggestions": []
        }
    
    # Log found parameters
    logger.info(f"Found {len(parameters)} parameters in the SCAD code")
    
    # Ask user if they want to provide specific parameter adjustments
    print("\n=== Parameter Tuning ===")
    print(f"Found {len(parameters)} parameters in the generated model:")
    
    # Display parameters in a nicely formatted way
    for i, (param_name, param_info) in enumerate(parameters.items(), 1):
        value = param_info["value"]
        param_type = param_info["type"]
        comment = f" // {param_info['comment']}" if param_info.get("comment") else ""
        
        print(f"{i}. {param_name} = {value} ({param_type}){comment}")
    
    # Ask user if they want to provide custom adjustments
    user_choice = input("\nWould you like to adjust any parameters manually? (yes/no): ").strip().lower()
    
    if user_choice in ["yes", "y"]:
        # User wants to provide manual adjustments
        user_adjustments = {}
        
        while True:
            param_to_adjust = input("\nEnter parameter name or number to adjust (or 'done' to finish): ").strip()
            
            if param_to_adjust.lower() in ["done", "exit", "quit", "q"]:
                break
            
            # Handle numeric input (converting to parameter name)
            if param_to_adjust.isdigit():
                param_idx = int(param_to_adjust) - 1
                if 0 <= param_idx < len(parameters):
                    param_to_adjust = list(parameters.keys())[param_idx]
                else:
                    print(f"Invalid parameter number. Please enter a number between 1 and {len(parameters)}")
                    continue
            
            # Check if parameter exists
            if param_to_adjust not in parameters:
                print(f"Parameter '{param_to_adjust}' not found. Please try again.")
                continue
            
            # Get current parameter info
            current_value = parameters[param_to_adjust]["value"]
            param_type = parameters[param_to_adjust]["type"]
            
            # Display current value and ask for new value
            print(f"Current value: {current_value} ({param_type})")
            
            if param_type == "vector":
                # For vectors, handle each component
                new_value = []
                
                for i, component in enumerate(current_value):
                    component_input = input(f"Enter new value for component {i+1} (or press Enter to keep {component}): ")
                    if component_input.strip():
                        try:
                            # Try to parse as appropriate numeric type
                            if isinstance(component, int):
                                new_component = int(component_input)
                            else:
                                new_component = float(component_input)
                            new_value.append(new_component)
                        except ValueError:
                            print(f"Invalid input for numeric value. Using original: {component}")
                            new_value.append(component)
                    else:
                        new_value.append(component)
            else:
                # For scalar values
                new_value_input = input(f"Enter new value for {param_to_adjust}: ")
                
                if new_value_input.strip():
                    try:
                        # Parse input according to parameter type
                        if param_type == "integer":
                            new_value = int(new_value_input)
                        elif param_type == "float":
                            new_value = float(new_value_input)
                        elif param_type == "boolean":
                            new_value = new_value_input.strip().lower() in ["true", "1", "yes", "y"]
                        else:
                            # For string or unknown types
                            new_value = new_value_input
                    except ValueError:
                        print(f"Invalid input for {param_type}. Using original value.")
                        new_value = current_value
                else:
                    # If empty input, keep original
                    print("No change made.")
                    continue
            
            # Add to adjustments dict
            user_adjustments[param_to_adjust] = {
                "old_value": current_value,
                "new_value": new_value,
                "type": param_type
            }
            print(f"Parameter '{param_to_adjust}' will be updated from {current_value} to {new_value}")
        
        return {
            "success": True,
            "user_provided": True,
            "parameters": parameters,
            "adjustments": user_adjustments
        }
    
    else:
        # User wants automatic suggestions - use LLM if provided
        if llm is None:
            logger.warning("No LLM provided for automatic parameter suggestions")
            return {
                "success": False,
                "error": "No LLM provided for automatic parameter suggestions",
                "parameters": parameters,
                "suggestions": []
            }
        
        print("\nGenerating parameter suggestions based on your description...")
        
        # Create a prompt for the LLM
        prompt = f"""
        Given this description: {description}
        
        And these parameters extracted from OpenSCAD code:
        {json.dumps({name: info for name, info in parameters.items()}, indent=2)}
        
        Please suggest adjustments to make the model better match the description.
        For each parameter that should be adjusted, provide:
        1. The parameter name
        2. The current value
        3. The suggested new value
        4. A brief explanation of why this change would improve the model
        
        Please format your response as a JSON object with this structure:
        {{
            "suggestions": [
                {{
                    "parameter": "parameter_name",
                    "current_value": current_value,
                    "suggested_value": new_value,
                    "explanation": "Explanation for the change"
                }},
                ...
            ]
        }}
        
        Focus on changes that would have a significant impact on the model's appearance or functionality.
        If some parameters are fine as they are, don't suggest changes for them.
        """
        
        # Generate suggestions using LLM
        try:
            response = llm.invoke(prompt)
            content = ""
            
            # Handle different response formats (streaming or not)
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # Extract JSON from response
            # Find JSON content between ``` markers or just extract the JSON object
            json_match = re.search(r'```(?:json)?\s*({\s*"suggestions":.+?})\s*```', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Try to find a JSON object without code block markers
                json_match = re.search(r'({[\s\S]*"suggestions"[\s\S]*})', content)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    # Fallback: try to use the entire response
                    json_content = content
            
            try:
                # Parse the JSON content
                suggestions = json.loads(json_content)
                
                # Return suggestions along with original parameters
                return {
                    "success": True,
                    "user_provided": False,
                    "parameters": parameters,
                    "suggestions": suggestions.get("suggestions", [])
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.debug(f"Raw response content: {content}")
                
                return {
                    "success": False,
                    "error": f"Failed to parse parameter suggestions: {str(e)}",
                    "parameters": parameters,
                    "raw_response": content
                }
        
        except Exception as e:
            logger.error(f"Error generating parameter suggestions: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating parameter suggestions: {str(e)}",
                "parameters": parameters
            }

def apply_parameter_adjustments(scad_code: str, adjustments: Dict[str, Dict[str, Any]]) -> str:
    """
    Apply parameter adjustments to SCAD code.
    
    Args:
        scad_code: Original OpenSCAD code
        adjustments: Dictionary of parameter adjustments
        
    Returns:
        Updated OpenSCAD code with adjustments applied
    """
    if not adjustments:
        logger.warning("No adjustments to apply")
        return scad_code
    
    logger.info(f"Applying {len(adjustments)} parameter adjustments")
    
    # Create a copy of the code to modify
    updated_code = scad_code
    
    for param_name, adjustment in adjustments.items():
        old_value = adjustment["old_value"]
        new_value = adjustment["new_value"]
        param_type = adjustment["type"]
        
        # Format new value according to its type
        if param_type == "string":
            new_value_str = f'"{new_value}"'
        elif param_type == "vector":
            new_value_str = f"[{', '.join(str(x) for x in new_value)}]"
        elif param_type == "boolean":
            new_value_str = "true" if new_value else "false"
        else:
            new_value_str = str(new_value)
        
        # Find and replace parameter declaration
        # Pattern to match parameter declaration
        pattern = rf'(^|\n)\s*({re.escape(param_name)})\s*=\s*([^;]+);'
        
        # Replace parameter declaration
        updated_code = re.sub(pattern, r'\1\2 = ' + new_value_str + ';', updated_code)
        
        # Also look for the parameter in module declarations
        module_pattern = rf'(module\s+[\w_]+\s*\([^)]*?)({re.escape(param_name)}\s*=\s*[^,\)]+)([,\)])'
        
        # Replace parameter in module declarations
        updated_code = re.sub(module_pattern, r'\1' + param_name + ' = ' + new_value_str + r'\3', updated_code)
    
    return updated_code

class ParameterTuner:
    """Class for managing parameter tuning for OpenSCAD models"""
    
    def __init__(self, llm=None):
        """
        Initialize the parameter tuner.
        
        Args:
            llm: Optional LLM instance for generating parameter suggestions
        """
        self.llm = llm
        logger.info("Parameter tuner initialized")
    
    def tune_parameters(self, scad_code: str, description: str) -> Dict[str, Any]:
        """
        Tune parameters for an OpenSCAD model.
        
        Args:
            scad_code: OpenSCAD code to tune
            description: Description of the desired model
            
        Returns:
            Dictionary containing tuning results
        """
        # Get parameter suggestions
        suggestions = suggest_parameter_adjustments(scad_code, description, self.llm)
        
        # If suggestions were automatically generated, ask user to review them
        if suggestions.get("success", False) and not suggestions.get("user_provided", True):
            auto_suggestions = suggestions.get("suggestions", [])
            
            if auto_suggestions:
                print("\n=== Suggested Parameter Adjustments ===")
                
                for idx, suggestion in enumerate(auto_suggestions, 1):
                    param = suggestion["parameter"]
                    current = suggestion["current_value"]
                    suggested = suggestion["suggested_value"]
                    explanation = suggestion["explanation"]
                    
                    print(f"\n{idx}. Parameter: {param}")
                    print(f"   Current: {current}")
                    print(f"   Suggested: {suggested}")
                    print(f"   Reason: {explanation}")
                
                # Ask user which suggestions to apply
                print("\nWhich suggestions would you like to apply?")
                print("Enter numbers separated by commas, 'all' for all suggestions, or 'none' to skip")
                user_choice = input("Your choice: ").strip().lower()
                
                if user_choice == "all":
                    # Apply all suggestions
                    adjustments = {}
                    for suggestion in auto_suggestions:
                        param = suggestion["parameter"]
                        adjustments[param] = {
                            "old_value": suggestions["parameters"][param]["value"],
                            "new_value": suggestion["suggested_value"],
                            "type": suggestions["parameters"][param]["type"]
                        }
                elif user_choice not in ["none", "skip", "cancel"]:
                    # Apply selected suggestions
                    try:
                        selected_indices = [int(idx.strip()) - 1 for idx in user_choice.split(",")]
                        adjustments = {}
                        
                        for idx in selected_indices:
                            if 0 <= idx < len(auto_suggestions):
                                suggestion = auto_suggestions[idx]
                                param = suggestion["parameter"]
                                adjustments[param] = {
                                    "old_value": suggestions["parameters"][param]["value"],
                                    "new_value": suggestion["suggested_value"],
                                    "type": suggestions["parameters"][param]["type"]
                                }
                    except ValueError:
                        print("Invalid input. No changes will be applied.")
                        adjustments = {}
                else:
                    # Skip all suggestions
                    adjustments = {}
                
                # Apply selected adjustments
                if adjustments:
                    updated_code = apply_parameter_adjustments(scad_code, adjustments)
                    return {
                        "success": True,
                        "original_code": scad_code,
                        "updated_code": updated_code,
                        "adjustments": adjustments
                    }
                else:
                    return {
                        "success": True,
                        "original_code": scad_code,
                        "updated_code": scad_code,  # No changes
                        "adjustments": {}
                    }
            else:
                print("\nNo parameter adjustments suggested.")
                return {
                    "success": True,
                    "original_code": scad_code,
                    "updated_code": scad_code,  # No changes
                    "adjustments": {}
                }
        
        # Handle user-provided adjustments
        elif suggestions.get("success", False) and suggestions.get("user_provided", False):
            adjustments = suggestions.get("adjustments", {})
            
            if adjustments:
                updated_code = apply_parameter_adjustments(scad_code, adjustments)
                return {
                    "success": True,
                    "original_code": scad_code,
                    "updated_code": updated_code,
                    "adjustments": adjustments
                }
            else:
                return {
                    "success": True,
                    "original_code": scad_code,
                    "updated_code": scad_code,  # No changes
                    "adjustments": {}
                }
        
        # Handle errors
        else:
            error = suggestions.get("error", "Unknown error in parameter tuning")
            logger.error(error)
            return {
                "success": False,
                "error": error,
                "original_code": scad_code,
                "updated_code": scad_code  # No changes
            }
