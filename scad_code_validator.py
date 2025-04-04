import re
from typing import Tuple, List

def validate_scad_code(code: str) -> Tuple[bool, List[str]]:
    """
    Validate OpenSCAD code for syntax and structure.
    
    Args:
        code: The OpenSCAD code to validate
        
    Returns:
        Tuple containing:
        - Boolean indicating if code is valid
        - List of validation messages/errors
    """
    messages = []
    
    # Check if code is empty
    if not code or len(code.strip()) == 0:
        messages.append("Error: OpenSCAD code is empty")
        return False, messages
    
    # Check code length
    if len(code) < 50:
        messages.append(f"Warning: Code is very short ({len(code)} characters, recommended minimum: 50)")
    
    if len(code) > 20000:
        messages.append(f"Error: Code exceeds maximum length ({len(code)} characters, maximum: 20000)")
        return False, messages
    
    # Check for balanced brackets and parentheses
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []
    
    for i, char in enumerate(code):
        if char in brackets.keys():
            stack.append((char, i))
        elif char in brackets.values():
            if not stack:
                messages.append(f"Error: Unmatched closing bracket '{char}' at position {i}")
                return False, messages
            
            last_open, last_pos = stack.pop()
            if char != brackets[last_open]:
                messages.append(f"Error: Mismatched brackets: '{last_open}' at position {last_pos} and '{char}' at position {i}")
                return False, messages
    
    if stack:
        positions = ', '.join([f"'{b}' at position {p}" for b, p in stack])
        messages.append(f"Error: Unclosed brackets: {positions}")
        return False, messages
    
    # Check for basic OpenSCAD elements
    basic_elements = {
        'module': re.compile(r'module\s+\w+\s*\('),
        'cube': re.compile(r'cube\s*\('),
        'sphere': re.compile(r'sphere\s*\('),
        'cylinder': re.compile(r'cylinder\s*\('),
        'translate': re.compile(r'translate\s*\('),
        'rotate': re.compile(r'rotate\s*\('),
        'union': re.compile(r'union\s*\('),
        'difference': re.compile(r'difference\s*\('),
        'intersection': re.compile(r'intersection\s*\(')
    }
    
    found_elements = []
    for element, pattern in basic_elements.items():
        if pattern.search(code):
            found_elements.append(element)
    
    if not found_elements:
        messages.append("Error: No basic OpenSCAD elements found. Code should include elements like cube, sphere, cylinder, etc.")
        return False, messages
    else:
        messages.append(f"Found OpenSCAD elements: {', '.join(found_elements)}")
    
    # Check for commented-out code
    comment_lines = len(re.findall(r'^\s*\/\/', code, re.MULTILINE))
    code_lines = len(code.strip().split('\n'))
    
    if comment_lines > code_lines / 2:
        messages.append(f"Warning: High proportion of commented code ({comment_lines} comment lines out of {code_lines} total lines)")
    
    # Check for variable definitions
    variable_defs = re.findall(r'^\s*([a-zA-Z0-9_]+)\s*=\s*([^;]+);', code, re.MULTILINE)
    if variable_defs:
        messages.append(f"Found {len(variable_defs)} variable definitions")
    else:
        messages.append("Warning: No variable definitions found. Consider using variables for parametric design.")
    
    # Check for potential errors
    potential_errors = [
        (r'[^\/]\/[^\s\/]', "Potential division operator missing spaces"),
        (r'[^=!<>]=[^=]', "Single equals sign used (assignment) where comparison might be intended"),
        (r'\w+\s*\(\s*\)', "Empty function/module parameters"),
        (r';\s*;', "Double semicolon"),
        (r'else\s*\w', "Missing space or braces after 'else'")
    ]
    
    for pattern, message in potential_errors:
        if re.search(pattern, code):
            messages.append(f"Warning: {message}")
    
    # Final validation result
    is_valid = not any("Error:" in msg for msg in messages)
    
    return is_valid, messages
