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
    
    if len(code) > 30000:
        messages.append(f"Error: Code exceeds maximum length ({len(code)} characters, maximum: 20000)")
        return False, messages
    
    # Check for balanced brackets and parentheses with detailed position tracking
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []
    
    for i, char in enumerate(code):
        if char in brackets.keys():
            stack.append((char, i))
        elif char in brackets.values():
            if not stack:
                line_num = code[:i].count('\n') + 1
                col_num = i - code[:i].rfind('\n')
                messages.append(f"Error: Unmatched closing bracket '{char}' at line {line_num}, column {col_num}")
                return False, messages
            
            last_open, last_pos = stack.pop()
            if char != brackets[last_open]:
                last_line = code[:last_pos].count('\n') + 1
                last_col = last_pos - code[:last_pos].rfind('\n')
                curr_line = code[:i].count('\n') + 1
                curr_col = i - code[:i].rfind('\n')
                messages.append(f"Error: Mismatched brackets: '{last_open}' at line {last_line}, column {last_col} and '{char}' at line {curr_line}, column {curr_col}")
                return False, messages
    
    if stack:
        remaining_info = []
        for bracket, pos in stack:
            line_num = code[:pos].count('\n') + 1
            col_num = pos - code[:pos].rfind('\n')
            remaining_info.append(f"'{bracket}' at line {line_num}, column {col_num}")
        
        messages.append(f"Error: Unclosed brackets: {', '.join(remaining_info)}")
        return False, messages
        
    # Check for missing semicolons
    missing_semicolons = check_missing_semicolons(code)
    if missing_semicolons:
        for line_num, line_text in missing_semicolons:
            messages.append(f"Warning: Possible missing semicolon at line {line_num}: '{line_text}'")
    
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

def check_missing_semicolons(code: str) -> List[Tuple[int, str]]:
    """
    Check for lines that might be missing semicolons in OpenSCAD code.
    
    Args:
        code: The OpenSCAD code to check
        
    Returns:
        List of tuples containing (line_number, line_text) for lines potentially missing semicolons
    """
    lines = code.split('\n')
    missing_semicolons = []
    
    # These are keywords that don't require semicolons at the end of the line
    no_semicolon_keywords = ['module', 'function', 'if', 'else', 'for', 'let']
    
    # Characters that indicate a line doesn't need a semicolon when it ends with them
    no_semicolon_endings = ['{', '}', '(', ';']
    
    for i, line in enumerate(lines):
        line_text = line.strip()
        
        # Skip empty lines or comment-only lines
        if not line_text or line_text.startswith('//'):
            continue
        
        # Skip multiline comments
        if line_text.startswith('/*') or line_text.endswith('*/'):
            continue
            
        # Skip lines ending with characters that don't need semicolons
        if any(line_text.endswith(char) for char in no_semicolon_endings):
            continue
            
        # Skip lines starting with keywords that don't need semicolons
        if any(line_text.startswith(keyword) for keyword in no_semicolon_keywords):
            continue
            
        # Skip lines with brackets/braces at the end (like "translate([0,0,0])" on its own line)
        if re.search(r'\)\s*$', line_text) and not is_complete_statement(line_text):
            continue
            
        # Check if next line is a continuation
        if (i < len(lines) - 1 and 
            lines[i+1].strip() and 
            lines[i+1].lstrip().startswith((')', ']', '}'))):
            continue
        
        # If we get here, this line might need a semicolon
        missing_semicolons.append((i + 1, line_text))
    
    return missing_semicolons

def is_complete_statement(line: str) -> bool:
    """
    Determine if a line represents a complete statement that would need a semicolon.
    
    Args:
        line: The line of code to check
        
    Returns:
        Boolean indicating if this is a complete statement
    """
    # This is a simplification - real parsing would be more complex
    # We're looking for lines that have balanced brackets and likely represent a complete statement
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    
    for char in line:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False  # Unbalanced brackets
            if char != brackets[stack.pop()]:
                return False  # Mismatched brackets
    
    # If brackets are balanced and it looks like an assignment or function call,
    # it's likely a complete statement
    return (not stack and 
            (re.search(r'=\s*\w+', line) or  # Assignment
             re.search(r'\w+\s*\([^()]*\)', line)))  # Function call

def validate_scad_syntax(code: str) -> bool:
    """
    Validate OpenSCAD syntax by checking for common errors.
    
    Args:
        code: The OpenSCAD code to validate
        
    Returns:
        Boolean indicating if the syntax is valid
    """
    # Check for mismatched braces
    if code.count('{') != code.count('}'):
        return False
        
    # Check for mismatched parentheses
    if code.count('(') != code.count(')'):
        return False
        
    # Check for mismatched square brackets
    if code.count('[') != code.count(']'):
        return False
    
    # Check for missing semicolons (simplified check)
    # We count non-comment, non-empty lines and compare with semicolon count
    # This is a heuristic and won't be perfect
    non_comment_lines = len([line for line in code.split('\n') 
                            if line.strip() and not line.strip().startswith('//')])
    semicolons = code.count(';')
    
    # If there are significantly fewer semicolons than non-comment lines,
    # we might have missing semicolons
    if semicolons < non_comment_lines / 2:
        return False
    
    return True