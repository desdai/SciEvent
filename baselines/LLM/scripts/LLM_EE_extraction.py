#!/usr/bin/env python3
"""
LLM Argument Extractor - Processes raw LLM output files to extract argument structures.
Handles JSON extraction from raw output files and merges them back into the original paper structure.
Includes fixes for common LLM JSON errors like incorrect <NONE> formatting, commas,
and invalid {"string"} formats instead of "string".
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(log_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("argument_extractor")


def fix_json_comprehensive(json_str: str, logger: logging.Logger, aggressive: bool = False) -> str:
    """
    Comprehensive JSON fix function that combines all fixing logic.
    
    Args:
        json_str: The JSON string to fix
        logger: Logger instance
        aggressive: Whether to use more aggressive fixing approaches
    
    Returns:
        Fixed JSON string
    """
    # Store original for comparison
    original_json = json_str
    
    try:
        # --- Handle invalid escape sequences ---
        json_str = json_str.replace("\\'", "'")
        
        # --- Fix empty arrays ---
        # Replace empty arrays with ["<NONE>"]
        json_str = re.sub(r'"\s*:\s*\[\s*\]', '": ["<NONE>"]', json_str)
        
        # Fix Object structure - ensure all four fields are present
        object_pattern = r'"Object"\s*:\s*\{([^}]*)\}'
        object_match = re.search(object_pattern, json_str)
        
        if object_match:
            object_content = object_match.group(1)
            
            # Check for missing fields and add them
            required_fields = ["Primary Object",  "Secondary Object"]
            for field in required_fields:
                if f'"{field}"' not in object_content:
                    # If any required field is missing, add it
                    if object_content.strip().endswith("}"):
                        # If Object content ends with closing brace, add before it
                        json_str = re.sub(
                            r'("Object"\s*:\s*\{[^}]*)(})', 
                            f'\\1,\n    "{field}": ["<NONE>"]\\2', 
                            json_str
                        )
                    else:
                        # If there's already content, add after the last field
                        json_str = re.sub(
                            r'("Object"\s*:\s*\{.*?)(?=\})', 
                            f'\\1,\n    "{field}": ["<NONE>"]', 
                            json_str,
                            flags=re.DOTALL
                        )
        
        # --- Fix unquoted <NONE> values ---
        fields_to_check = ["Method", "Analysis", "Challenge", "Ethical", 
                         "Implications", "Contradictions", "Purpose", "Results", 
                         "Agent", "Context", "Primary Object",
                         "Secondary Object"]
        
        for field in fields_to_check:
            # Fix pattern where <NONE> appears without quotes in an array
            pattern = f'"{field}"\\s*:\\s*\\[\\s*<NONE>\\s*\\]'
            json_str = re.sub(pattern, f'"{field}": ["<NONE>"]', json_str)
            
            # Also handle other variations
            alt_pattern = f'"{field}"\\s*:\\s*<NONE>'
            json_str = re.sub(alt_pattern, f'"{field}": ["<NONE>"]', json_str)
            
            # Handle case-insensitive <none> if aggressive
            if aggressive:
                ignore_case_pattern = f'"{field}"\\s*:\\s*\\[\\s*<none>\\s*\\]'
                json_str = re.sub(ignore_case_pattern, f'"{field}": ["<NONE>"]', json_str, flags=re.IGNORECASE)
                
                # Handle <NONE> without brackets at all
                no_brackets_pattern = f'"{field}"\\s*:\\s*<NONE>\\s*,'
                json_str = re.sub(no_brackets_pattern, f'"{field}": ["<NONE>"],', json_str, flags=re.IGNORECASE)
                
                # Handle at the end of JSON
                end_pattern = f'"{field}"\\s*:\\s*<NONE>\\s*}}'
                json_str = re.sub(end_pattern, r'"{field}": ["<NONE>"]}}'.format(field=field), json_str, flags=re.IGNORECASE)
        
        # General fix for unquoted <NONE> values
        json_str = re.sub(r'\[\s*<NONE>\s*\]', '["<NONE>"]', json_str)
        # Fix for unquoted values not in arrays
        json_str = re.sub(r':\s*<NONE>', ': "<NONE>"', json_str)
        
        # --- Fix mismatched quotes ---
        # Fix single quotes used for string values in arrays
        array_pattern = r'"([^"]+)"\s*:\s*\[\s*"([^"]*?)\'(\s*)\]'
        json_str = re.sub(array_pattern, r'"\1": ["\2"\3]', json_str)
        
        # Fix object fields with single quotes or missing quotes
        object_fields = ["Primary Object", "Secondary Object"]
        for field in object_fields:
            # Fix "Field": ["value'] -> "Field": ["value"]
            pattern = f'"{field}"\\s*:\\s*\\[\\s*"([^"]*?)\'(\\s*)\\]'
            json_str = re.sub(pattern, f'"{field}": ["\\1"\\2]', json_str)
            
            # Fix missing closing quote: "Field": ["value] -> "Field": ["value"]
            missing_quote_pattern = f'"{field}"\\s*:\\s*\\[\\s*"([^"]*?)(\\s*)\\](?!")'
            json_str = re.sub(missing_quote_pattern, f'"{field}": ["\\1"\\2]', json_str)
            
            # Fix single quotes entirely: "Field": ['value'] -> "Field": ["value"]
            single_quotes_pattern = f'"{field}"\\s*:\\s*\\[\\s*\'([^\']*?)\'(\\s*)\\]'
            json_str = re.sub(single_quotes_pattern, f'"{field}": ["\\1"\\2]', json_str)
        
        # Fix general array fields with single quotes
        other_fields = ["Agent", "Context", "Purpose", "Method", "Results", 
                        "Analysis", "Challenge", "Ethical", "Implications", "Contradictions"]
        for field in other_fields:
            # Fix "Field": ["value'] -> "Field": ["value"]
            pattern = f'"{field}"\\s*:\\s*\\[\\s*"([^"]*?)\'(\\s*)\\]'
            json_str = re.sub(pattern, f'"{field}": ["\\1"\\2]', json_str)
            
            # Fix "Field": ['value'] -> "Field": ["value"]
            single_quotes_pattern = f'"{field}"\\s*:\\s*\\[\\s*\'([^\']*?)\'(\\s*)\\]'
            json_str = re.sub(single_quotes_pattern, f'"{field}": ["\\1"\\2]', json_str)
            
            # Fix "Field": ["value] -> "Field": ["value"]
            missing_quote_pattern = f'"{field}"\\s*:\\s*\\[\\s*"([^"]*?)(\\s*)\\](?!")'
            json_str = re.sub(missing_quote_pattern, f'"{field}": ["\\1"\\2]', json_str)
        
        if aggressive:
            # Fix 'field': ['value'] -> "field": ["value"]
            json_str = re.sub(r"'([^']+)'\\s*:\\s*\\[\\s*'([^']*?)'\\s*\\]", r'"\1": ["\2"]', json_str)
            
            # Fix "field': ["value"] -> "field": ["value"]
            json_str = re.sub(r'"([^"]+)\'\\s*:', r'"\1":', json_str)
            
            # Fix 'field": ["value"] -> "field": ["value"]
            json_str = re.sub(r"'([^']+)\"\\s*:", r'"\1":', json_str)
            
            # Fix remaining single quotes if likely a problem
            has_value_with_single_quote = re.search(r':\s*\[\s*\'[^\']*\'', json_str)
            if has_value_with_single_quote:
                json_str = re.sub(r":\s*\[\s*'([^']*?)'\s*\]", r': ["\1"]', json_str)
        
        # --- Fix nested quoted arrays ---
        fields_to_fix = ["Method", "Analysis", "Challenge", "Ethical", 
                         "Implications", "Contradictions", "Purpose", "Results"]
        
        for field in fields_to_fix:
            # Pattern 1: "Field": ["["<NONE>"]"]
            pattern1 = f'"{field}":\\s*\\[\\s*"\\[\\s*\\"<NONE>\\"\\s*\\]"\\s*\\]'
            json_str = re.sub(pattern1, f'"{field}": ["<NONE>"]', json_str)
            
            # Pattern 2: "Field": ["[""<NONE>""]"]
            pattern2 = f'"{field}":\\s*\\[\\s*"\\[\\"\\"<NONE>\\"\\"\\]"\\s*\\]'
            json_str = re.sub(pattern2, f'"{field}": ["<NONE>"]', json_str)
            
            # Pattern 3: "Field": ["[""<NONE>"']"]
            pattern3 = f'"{field}":\\s*\\[\\s*"\\[\\"\\"<NONE>\\"\\\'\\]"\\s*\\]'
            json_str = re.sub(pattern3, f'"{field}": ["<NONE>"]', json_str)
            
            # Pattern 4: More generally looking for ["[...something...]"]
            generic_pattern = f'"{field}":\\s*\\[\\s*"\\[(.+?)\\]"\\s*\\]'
            
            # Find all matches
            for match in re.finditer(generic_pattern, json_str):
                full_match = match.group(0)
                inner_content = match.group(1)
                
                # Extract the actual content, removing any extra quotes
                inner_text = inner_content.replace('"', '').replace("'", '').strip()
                
                # If it's just <NONE>, replace with standard format
                if inner_text.upper() == "<NONE>":
                    replacement = f'"{field}": ["<NONE>"]'
                else:
                    # Otherwise keep the actual content
                    replacement = f'"{field}": ["{inner_text}"]'
                
                json_str = json_str.replace(full_match, replacement)
        
        # General cleanup of nested quotes in arrays across all fields
        if aggressive:
            # Find all array fields and fix them
            array_pattern = r'"([^"]+)"\s*:\s*\[(.*?)\]'
            
            for match in re.finditer(array_pattern, json_str, re.DOTALL):
                field_name = match.group(1)
                array_content = match.group(2)
                
                # Skip if no nested quotes or brackets
                if '[' not in array_content and ']' not in array_content:
                    continue
                
                # Get the full match
                full_match = f'"{field_name}": [{array_content}]'
                
                # Look for nested array patterns: ["[...]"]
                nested_pattern = r'"(\[.*?\])"'
                
                if re.search(nested_pattern, array_content):
                    # Extract items from the array
                    items = []
                    for item_match in re.finditer(r'"(\[.*?\])"', array_content):
                        nested_array = item_match.group(1)
                        # Extract content from inner array
                        inner_match = re.search(r'\[\s*"([^"]*)"(?:\s*,\s*"([^"]*)")*\s*\]', nested_array)
                        if inner_match:
                            inner_content = inner_match.group(1)
                            items.append(f'"{inner_content}"')
                        else:
                            # Fallback if we can't parse inner array
                            items.append('"<NONE>"')
                    
                    # Special handling for <NONE> cases
                    if len(items) == 1 and ("<NONE>" in items[0] or items[0] == '""'):
                        replacement = f'"{field_name}": ["<NONE>"]'
                    else:
                        replacement = f'"{field_name}": [{", ".join(items)}]'
                    
                    json_str = json_str.replace(full_match, replacement)
            
            # More aggressive handling of nested quotes for specific fields
            fields_for_aggressive = ["Analysis", "Secondary Object", "Primary Object", "Method", "Results"]
            for match in re.finditer(array_pattern, json_str, re.DOTALL):
                field_name = match.group(1)
                array_content = match.group(2)
                
                # For fields that often contain text with nested quotes
                if field_name in fields_for_aggressive:
                    # Replace all internal quotes with single quotes
                    fixed_content = array_content.replace('"', "'")
                    # Preserve the outer quotes for array items
                    fixed_content = re.sub(r"'([^']*)'", r'"\1"', fixed_content)
                    
                    # Replace in the JSON
                    original = f'"{field_name}": [{array_content}]'
                    fixed = f'"{field_name}": [{fixed_content}]'
                    json_str = json_str.replace(original, fixed)
        
        # --- Fix double-quoted <NONE> values ---
        # Fix: ""<NONE>"" -> "<NONE>"
        json_str = re.sub(r'""<NONE>""', '"<NONE>"', json_str)
        
        # Fix: [""]"<NONE>"""] -> ["<NONE>"]
        json_str = re.sub(r'\["*"<NONE>"*"\]', '["<NONE>"]', json_str)
        
        # Fix for single items with extra quotes: [""text""] -> ["text"]
        json_str = re.sub(r'\["+"([^"]+)"+"]\]', r'["\1"]', json_str)
        
        # --- Fix Context array with mixed <NONE> values ---
        context_pattern = r'"Context":\s*\[(.*?)\]'
        context_match = re.search(context_pattern, json_str, re.DOTALL)
        
        if context_match:
            context_content = context_match.group(1)
            
            # Fix double-quoted strings in Context array
            fixed_context = re.sub(r'"+"([^"]+)"+"', r'"\1"', context_content)
            
            # Replace in the JSON
            json_str = json_str.replace(f'"Context": [{context_content}]', f'"Context": [{fixed_context}]')
        
        # --- Fix for invalid {"string"} format ---
        # Try repeatedly in case of multiple occurrences
        previous_json = ""
        while previous_json != json_str:
            previous_json = json_str
            # Regex: \{ " (capture non-quote chars) " \} -> " captured_chars "
            json_str = re.sub(r'\{"([^"]+)"\}', r'"\1"', json_str)
        
        # --- Fix for nested/duplicated Object issue ---
        # Fix for: "Primary Object": "Object": { ... instead of "Primary Object": "value"
        nested_obj_pattern = r'"Primary Object"\s*:\s*"Object"\s*:\s*\{'
        if re.search(nested_obj_pattern, json_str):
            # Get the actual correct Primary Object value by looking for what it should be
            primary_obj_match = re.search(r'"Primary Object"\s*:\s*"Object"\s*:\s*\{\s*"Primary Object"\s*:\s*"([^"]+)"', json_str)
            if primary_obj_match:
                correct_value = primary_obj_match.group(1)
                # Replace the corrupted structure with the correct value
                json_str = re.sub(
                    r'"Primary Object"\s*:\s*"Object"\s*:\s*\{\s*"Primary Object"\s*:\s*"[^"]+"\s*\}',
                    f'"Primary Object": "{correct_value}"',
                    json_str
                )
        
        # --- Fix for misplaced Object fields ---
        # Match object section that ends prematurely
        object_fix_pattern = r'("Object"\s*:\s*\{[^{}]*?\})\s*,\s*("Secondary Object"\s*:.*?)\s*\},'
        # Replace with correctly structured object
        json_str = re.sub(object_fix_pattern, r'"Object": {\n    "Primary Object": \1,\n    \2,\n    \3\n  },', json_str)
        
        # Alternative pattern with double closing braces
        alt_pattern = r'"Object"\s*:\s*\{\s*"Primary Object"\s*:\s*"([^"]+)"\s*,\s*"Secondary Object"\s*:\s*(\[[^\]]+\])\s*(\[[^\]]+\])'
        replacement = r'"Object": {\n    "Primary Object": "\1",\n    "Secondary Object": \2,\n }'
        json_str = re.sub(alt_pattern, replacement, json_str)
        
        # --- Fix for duplicate Primary Object entries ---
        duplicate_primary_obj_pattern = r'"Primary Object"\s*:.*?,.*?"Primary Object"\s*:'
        if re.search(duplicate_primary_obj_pattern, json_str):
            # Extract the entire Object structure
            object_pattern = r'"Object"\s*:\s*\{([\s\S]*?)\}'
            object_match = re.search(object_pattern, json_str)
            
            if object_match:
                object_content = object_match.group(1)
                
                # Find all Primary Object entries
                primary_obj_entries = re.findall(r'"Primary Object"\s*:\s*(\[[^\]]*\]|\[[^\]]*\]|\{[^}]*\}|"[^"]*")', object_content)
                
                if primary_obj_entries and len(primary_obj_entries) > 1:
                    # Use the first non-empty Primary Object entry
                    valid_entry = None
                    for entry in primary_obj_entries:
                        if entry and entry != '[]' and entry != '["<NONE>"]' and entry != '"<NONE>"':
                            valid_entry = entry
                            break
                    
                    if not valid_entry:
                        valid_entry = primary_obj_entries[0]  # Use first entry if all are empty
                    
                    # Create a fixed Object structure with only one Primary Object
                    fixed_object = f'"Object": {{\n    "Primary Object": {valid_entry},\n'
                    
                    
                    # Add Secondary Object if present
                    sec_obj_match = re.search(r'"Secondary Object"\s*:\s*(\[[^\]]*\]|\{[^}]*\}|"[^"]*")', object_content)
                    if sec_obj_match:
                        fixed_object += f'    "Secondary Object": {sec_obj_match.group(1)},\n'
                    else:
                        fixed_object += '    "Secondary Object": ["<NONE>"],\n'
                    
                    
                    # Replace the original Object structure with our fixed version
                    json_str = re.sub(object_pattern, fixed_object, json_str)
        
        # --- Fix trailing commas ---
        json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
        
        # --- Fix missing commas between fields ---
        lines = json_str.split('\n')
        fixed_lines = []
        for i in range(len(lines)):
            current = lines[i].rstrip()
            fixed_lines.append(current) # Add current line first

            if i < len(lines) - 1: # Check if there is a next line
                next_line = lines[i+1].strip()

                # Conditions for adding a comma:
                # 1. Current line ends with a value (quote, bracket, brace)
                # 2. Current line does NOT already end with a comma
                # 3. Next line starts with a key (quote) or structure (brace, bracket)
                # 4. Next line isn't just closing a structure (avoids adding comma before final }/])
                ends_with_value = current.endswith('"') or current.endswith(']') or current.endswith('}')
                not_ends_with_comma = not current.endswith(',')
                next_line_starts_item = next_line.startswith('"') or next_line.startswith('{') or next_line.startswith('[')
                is_closing = next_line.startswith('}') or next_line.startswith(']')

                if ends_with_value and not_ends_with_comma and next_line_starts_item and not is_closing:
                     # Modify the *stored* line in fixed_lines
                     fixed_lines[-1] = current + ','
        
        json_str = '\n'.join(fixed_lines)
        
        # --- Final cleanup ---
        # One last fix for empty arrays
        json_str = re.sub(r'\[\s*\]', '["<NONE>"]', json_str)
        
        # Ensure all field arrays contain "<NONE>"
        for field in fields_to_fix:
            # Fix: "Field": [] -> "Field": ["<NONE>"]
            json_str = re.sub(f'"{field}":\\s*\\[\\s*\\]', f'"{field}": ["<NONE>"]', json_str)
            
            # Fix: "Field": [""] -> "Field": ["<NONE>"]
            json_str = re.sub(f'"{field}":\\s*\\[\\s*""\\s*\\]', f'"{field}": ["<NONE>"]', json_str)
        
        # Fix trailing commas one last time after possible additions
        json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
        
        # Log if changes were made
        if json_str != original_json:
            logger.debug(f"Fixed JSON issues. Aggressive mode: {aggressive}")
        
        return json_str
        
    except Exception as e:
        logger.warning(f"Error during fix_json_comprehensive: {e}")
        return json_str  # Return original on error


def format_argument_value(value: Any) -> List:
    """
    Format argument values according to requirements:
    - If value is "<NONE>" or None, return ["<NONE>"]
    - If value is already a list, clean it and return (or ["<NONE>"] if empty after clean)
    - Otherwise wrap single value in a list
    """
    if value is None or value == "<NONE>" or value == "NONE":
        return ["<NONE>"]

    if isinstance(value, list):
        # Check for empty list case first (important fix)
        if not value:
            return ["<NONE>"]  # Return ["<NONE>"] for empty lists
            
        # If list contains "<NONE>" or "NONE", replace with proper format
        formatted_list = []
        for item in value:
            if item == "<NONE>" or item == "NONE" or item is None:
                # Ensure only valid "<NONE>" string is added once if encountered
                if "<NONE>" not in formatted_list:
                    formatted_list.append("<NONE>")
            elif item: # Append non-empty items
                formatted_list.append(item)

        # If the list ended up empty or only contained NONE types, return ["<NONE>"]
        # Check if list contains anything other than "<NONE>"
        has_real_value = any(item != "<NONE>" for item in formatted_list)
        if not formatted_list or not has_real_value:
            return ["<NONE>"]
        else:
            # Remove potential duplicate "<NONE>" if real values exist
            if "<NONE>" in formatted_list and len(formatted_list) > 1:
                return [item for item in formatted_list if item != "<NONE>"]
            else:
                return formatted_list

    # Handle empty string case
    if value == "":
        return ["<NONE>"]
        
    # Wrap single non-list, non-None value in a list
    return [value]


def structure_arguments(parsed_json: Dict, logger: logging.Logger) -> Tuple[str, Dict]:
    """
    Structure the parsed JSON into the expected argument format.

    Args:
        parsed_json: The parsed JSON
        logger: Logger instance

    Returns:
        Tuple of Main Action and structured arguments
    """
    # Define the standard structure with default values
    arguments = {
        "Agent": ["<NONE>"],
        "Object": {
            "Primary Object": ["<NONE>"],
            "Secondary Object": ["<NONE>"]
        },
        "Context": ["<NONE>"],
        "Purpose": ["<NONE>"],
        "Method": ["<NONE>"],
        "Results": ["<NONE>"],
        "Analysis": ["<NONE>"],
        "Challenge": ["<NONE>"],
        "Ethical": ["<NONE>"],
        "Implications": ["<NONE>"],
        "Contradictions": ["<NONE>"]
    }

    # Extract Main Action (not part of the arguments dict itself)
    main_action = parsed_json.get("Main Action", "<NONE>")
    # Ensure main_action is a string, handle None or list cases defensively
    if isinstance(main_action, list):
         main_action = main_action[0] if main_action else "<NONE>"
    elif main_action is None:
         main_action = "<NONE>"

    # Extract and format all top-level fields using the keys from the default structure
    for field in arguments.keys():
        if field == "Object": # Handle nested Object separately
            continue
        if field in parsed_json:
            arguments[field] = format_argument_value(parsed_json[field])
        else:
             arguments[field] = ["<NONE>"] # Ensure default if key missing entirely

    # Extract and format Object fields specifically
    object_data = parsed_json.get("Object")
    if isinstance(object_data, dict):
        for sub_field in arguments["Object"].keys():
            if sub_field in object_data:
                arguments["Object"][sub_field] = format_argument_value(object_data[sub_field])
            else:
                 arguments["Object"][sub_field] = ["<NONE>"] # Ensure default if sub-key missing
    else:
        # If "Object" key exists but is not a dict, or doesn't exist, keep default values
         logger.debug(f"Object field was not a dict or missing, using default.")
         pass # Defaults are already set

    return main_action, arguments


def extract_json_from_raw_output(raw_output_path: str, logger: logging.Logger) -> Tuple[Optional[Dict], str]:
    """
    Extract valid JSON data from raw output files, trying all potential JSON structures.
    If one structure fails, tries the next one until a valid JSON is found.

    Args:
        raw_output_path: Path to the raw output file
        logger: Logger instance

    Returns:
        Tuple containing:
        - Extracted JSON data as a dictionary, or None if all extraction attempts fail
        - Error type string describing the failure reason if any
    """
    try:
        # Read the raw output file
        with open(raw_output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Prepare debug directory and file path
        debug_dir = os.path.join(os.path.dirname(raw_output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"{os.path.basename(raw_output_path)}.debug")
        
        # Find all potential JSON structures (from opening to closing braces)
        json_candidates = []
        start_indexes = [m.start() for m in re.finditer(r'(?<![a-zA-Z0-9_])\{', content)]
        
        # Log how many potential JSON structures were found
        logger.info(f"Found {len(start_indexes)} potential JSON structures in {raw_output_path}")
        
        # Extract each potential JSON structure
        for start_index in start_indexes:
            # Track nesting level to find matching closing brace
            level = 0
            in_string = False
            escape_next = False
            end_index = -1
            
            for i in range(start_index, len(content)):
                char = content[i]
                
                # Handle string boundaries and escaping
                if char == '\\' and not escape_next:
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                
                escape_next = False
                
                # Only count braces outside of strings
                if not in_string:
                    if char == '{':
                        level += 1
                    elif char == '}':
                        level -= 1
                        if level == 0:
                            end_index = i + 1
                            break
            
            if end_index > start_index:
                json_candidates.append(content[start_index:end_index])
        
        if not json_candidates:
            logger.warning(f"No complete JSON structures found in {raw_output_path}")
            # Save the problematic content to debug file
            with open(debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write(f"--- NO JSON STRUCTURES FOUND ---\n{content[:1000]}...\n\n")
            return None, "NO_JSON_FOUND"
            
        # Try parsing each JSON candidate with progressive fixing approaches
        for idx, json_str in enumerate(json_candidates):
            logger.info(f"Attempting to parse JSON candidate {idx+1}/{len(json_candidates)}")
            
            # Save original candidate for debugging
            try:
                with open(debug_file, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"--- JSON Candidate #{idx+1} ---\n{json_str}\n\n")
            except Exception as write_err:
                logger.error(f"Failed to write debug info: {write_err}")
                
            # STEP 1: Try parsing as-is first
            try:
                data = json.loads(json_str)
                logger.info(f"Successfully parsed JSON candidate #{idx+1} without fixes")
                return data, "SUCCESS"
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error on candidate #{idx+1}: {e}. Attempting fixes...")
                
                # STEP 2: Apply standard fixes
                try:
                    fixed_json = fix_json_comprehensive(json_str, logger, aggressive=False)
                    
                    try:
                        data = json.loads(fixed_json)
                        logger.info(f"Successfully parsed JSON candidate #{idx+1} after standard fixes")
                        return data, "SUCCESS_AFTER_STANDARD_FIXES"
                    except json.JSONDecodeError as e2:
                        logger.debug(f"Standard fixes failed for candidate #{idx+1}: {e2}")
                except Exception as fix_err:
                    logger.debug(f"Error during standard fixes for candidate #{idx+1}: {fix_err}")
                
                # STEP 3: Apply aggressive fixes
                try:
                    aggressive_json = fix_json_comprehensive(json_str, logger, aggressive=True)
                    
                    try:
                        data = json.loads(aggressive_json)
                        logger.info(f"Successfully parsed JSON candidate #{idx+1} after aggressive fixes")
                        return data, "SUCCESS_AFTER_AGGRESSIVE_FIXES"
                    except json.JSONDecodeError as e3:
                        logger.debug(f"Aggressive fixes failed for candidate #{idx+1}: {e3}")
                except Exception as agg_fix_err:
                    logger.debug(f"Error during aggressive fixes for candidate #{idx+1}: {agg_fix_err}")
                
                # Continue to the next candidate if all fixing attempts failed
                logger.warning(f"All fixes failed for JSON candidate #{idx+1}, trying next candidate")
        
        # If we've tried all candidates and none worked, attempt full reconstruction on the most promising one
        logger.warning(f"All {len(json_candidates)} JSON candidates failed to parse, attempting reconstruction")
        
        # Choose the most promising candidate (usually the longest one)
        best_candidate = max(json_candidates, key=len)
        
        # Try reconstructing from scratch
        try:
            # Extract fields from the best candidate
            summary_match = re.search(r'"Summary"\s*:\s*"([^"]+)"', best_candidate)
            summary = summary_match.group(1) if summary_match else "<NONE>"
            
            main_action_match = re.search(r'"Main Action"\s*:\s*"([^"]+)"', best_candidate)
            main_action = main_action_match.group(1) if main_action_match else "<NONE>"
            
            # Construct completely new valid JSON
            reconstructed_json = f"""{{
  "Summary": "{summary}",
  "Main Action": "{main_action}",
  "Agent": ["<NONE>"],
  "Object": {
    "Primary Object": ["<NONE>"],
    "Secondary Object": ["<NONE>"]
  },
  "Context": ["<NONE>"],
  "Purpose": ["<NONE>"],
  "Method": ["<NONE>"],
  "Results": ["<NONE>"],
  "Analysis": ["<NONE>"],
  "Challenge": ["<NONE>"],
  "Ethical": ["<NONE>"],
  "Implications": ["<NONE>"],
  "Contradictions": ["<NONE>"]
}}"""
            
            try:
                data = json.loads(reconstructed_json)
                logger.info(f"Successfully reconstructed JSON from best candidate")
                return data, "SUCCESS_AFTER_RECONSTRUCTION"
            except json.JSONDecodeError as recon_err:
                logger.error(f"Reconstructed JSON still has errors: {recon_err}")
                return None, "RECONSTRUCTION_FAILED"
        except Exception as extract_err:
            logger.error(f"Error during reconstruction attempt: {extract_err}")
            return None, "RECONSTRUCTION_ERROR"
        

    except FileNotFoundError:
        logger.error(f"Raw output file not found: {raw_output_path}")
        return None, "FILE_NOT_FOUND"
    except Exception as e:
        logger.error(f"Unexpected error processing {raw_output_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, "UNEXPECTED_ERROR"


def process_event(event_path: str, logger: logging.Logger, metrics: Dict) -> Tuple[Optional[str], Optional[Dict], Optional[str], bool, str]:
    """
    Process a single event file (raw LLM output) and extract arguments.

    Args:
        event_path: Path to the event file (e.g., paper_code_event_0.txt)
        logger: Logger instance
        metrics: Metrics dictionary to update

    Returns:
        Tuple containing:
        - Main Action (str, "ERROR", or "<NONE>")
        - Arguments dictionary (or None if extraction failed completely)
        - Summary (str, "ERROR", or "<NONE>")
        - has_error: Boolean indicating if a complete parsing error occurred
        - error_type: String indicating the type of error that occurred, if any
    """
    # Extract JSON from raw output, using the enhanced function with error type
    parsed_json, error_type = extract_json_from_raw_output(event_path, logger)

    if not parsed_json:
        logger.warning(f"Failed to extract and parse valid JSON from {event_path}: {error_type}")
        metrics["failed_extractions"] += 1  # Increment failed extractions counter
        
        # Track specific error types
        error_type_key = f"error_type_{error_type}"
        if error_type_key not in metrics:
            metrics[error_type_key] = 0
        metrics[error_type_key] += 1
        
        # Add file to appropriate failed files category
        if "failed_files_by_type" not in metrics:
            metrics["failed_files_by_type"] = {}
        if error_type not in metrics["failed_files_by_type"]:
            metrics["failed_files_by_type"][error_type] = []
        
        # Add file path to the specific error type list if not already there
        if event_path not in metrics["failed_files_by_type"][error_type]:
            metrics["failed_files_by_type"][error_type].append(event_path)
        
        # Also add to general failed files list for backward compatibility
        if event_path not in metrics["failed_files"]:
            metrics["failed_files"].append(event_path)
            
        # Return ERROR indicators for all values
        return "ERROR", None, "ERROR", True, error_type

    # If JSON parsing succeeded, proceed with standard processing
    metrics["successful_extractions"] += 1  # Increment successful extractions
    
    # Track success type for metrics
    success_type_key = f"success_type_{error_type}"  # error_type is actually success type here
    if success_type_key not in metrics:
        metrics[success_type_key] = 0
    metrics[success_type_key] += 1

    # Get summary - ensure it's a string
    # BUG FIX: Check that parsed_json is a dictionary before calling .get()
    if isinstance(parsed_json, dict):
        summary = parsed_json.get("Summary", "<NONE>")
        if isinstance(summary, list):
            summary = summary[0] if summary else "<NONE>"
        elif summary is None:
            summary = "<NONE>"
    else:
        # If somehow parsed_json is not a dict (should never happen with proper error handling)
        logger.error(f"Unexpected type for parsed_json: {type(parsed_json)}. Converting to string.")
        summary = str(parsed_json) if parsed_json else "<NONE>"
        # Force into error state
        return "ERROR", None, summary, True, "UNEXPECTED_TYPE"

    # Structure arguments using the extracted JSON
    try:
        main_action, arguments = structure_arguments(parsed_json, logger)
    except Exception as e:
        logger.error(f"Error structuring arguments: {e}")
        return "ERROR", None, summary, True, "STRUCTURE_ERROR"

    return main_action, arguments, summary, False, error_type  # No error


def process_paper(paper_code: str, abstract: str, events: List, raw_output_dir: str, logger: logging.Logger, metrics: Dict) -> Dict:
    """
    Process a single paper and its events, merging extracted arguments.

    Args:
        paper_code: The paper code (used for finding raw files)
        abstract: The paper's abstract text
        events: List of events/segments from the paper (can be empty)
        raw_output_dir: Directory containing raw LLM output files
        logger: Logger instance
        metrics: Metrics dictionary (passed to process_event)

    Returns:
        Processed paper dictionary with extracted arguments merged into events
    """
    logger.info(f"Processing paper: {paper_code}")
    processed_events = []

    # Define the default empty argument structure for cases where extraction fails
    default_arguments = {
        "Agent": ["<NONE>"], 
        "Object": {
            "Primary Object": ["<NONE>"], 
            "Secondary Object": ["<NONE>"]
        },
        "Context": ["<NONE>"], 
        "Purpose": ["<NONE>"], 
        "Method": ["<NONE>"], 
        "Results": ["<NONE>"], 
        "Analysis": ["<NONE>"],
        "Challenge": ["<NONE>"], 
        "Ethical": ["<NONE>"], 
        "Implications": ["<NONE>"], 
        "Contradictions": ["<NONE>"]
    }

    # Define the error argument structure for cases with complete parsing failure
    error_arguments = {
        "Agent": ["ERROR"], 
        "Object": {
            "Primary Object": ["ERROR"], 
            "Secondary Object": ["ERROR"]       },
        "Context": ["ERROR"], 
        "Purpose": ["ERROR"], 
        "Method": ["ERROR"], 
        "Results": ["ERROR"], 
        "Analysis": ["ERROR"],
        "Challenge": ["ERROR"], 
        "Ethical": ["ERROR"], 
        "Implications": ["ERROR"], 
        "Contradictions": ["ERROR"]
    }

    if not events:
        # Case 1: No events provided, process the full abstract
        logger.info(f"No events for {paper_code}, processing full abstract.")
        raw_file_path = os.path.join(raw_output_dir, f"{paper_code}_full.txt")

        if os.path.exists(raw_file_path):
            main_action, arguments, summary, has_error, error_type = process_event(raw_file_path, logger, metrics)

            # Track paper-level metrics
            if has_error:
                if "papers_with_errors" not in metrics:
                    metrics["papers_with_errors"] = 0
                metrics["papers_with_errors"] += 1
                
                if "papers_by_error_type" not in metrics:
                    metrics["papers_by_error_type"] = {}
                if error_type not in metrics["papers_by_error_type"]:
                    metrics["papers_by_error_type"][error_type] = []
                if paper_code not in metrics["papers_by_error_type"][error_type]:
                    metrics["papers_by_error_type"][error_type].append(paper_code)

            # Create a single event structure for the full abstract
            event = {
                "Text": abstract,
                "Main Action": main_action if not has_error else "ERROR",
                # Use extracted arguments if successful, use error structure if complete failure
                "Arguments": error_arguments if has_error else (arguments if arguments else default_arguments)
            }
            
            # Add summary if available and not None/<NONE>
            if summary and summary != "<NONE>":
                event["Summary"] = summary
                
            # Add error type information if an error occurred
            if has_error:
                event["Error_Type"] = error_type
                
            processed_events.append(event)
        else:
            logger.warning(f"Raw output file not found for full abstract: {raw_file_path}")
            # Create a default event structure if raw file is missing
            # Mark with a specific error type for missing files
            if "missing_raw_files" not in metrics:
                metrics["missing_raw_files"] = 0
            metrics["missing_raw_files"] += 1
            
            # Track this as a paper with errors
            if "papers_with_errors" not in metrics:
                metrics["papers_with_errors"] = 0
            metrics["papers_with_errors"] += 1
            
            # Add to papers by error type for missing files
            if "papers_by_error_type" not in metrics:
                metrics["papers_by_error_type"] = {}
            if "MISSING_RAW_FILE" not in metrics["papers_by_error_type"]:
                metrics["papers_by_error_type"]["MISSING_RAW_FILE"] = []
            if paper_code not in metrics["papers_by_error_type"]["MISSING_RAW_FILE"]:
                metrics["papers_by_error_type"]["MISSING_RAW_FILE"].append(paper_code)
            
            event = {
                "Text": abstract,
                "Main Action": "ERROR",
                "Arguments": error_arguments,
                "Error_Type": "MISSING_RAW_FILE"
            }
            processed_events.append(event)
    else:
        # Case 2: Process each event separately
        logger.info(f"Processing {len(events)} events for paper {paper_code}.")
        
        # Track if paper has at least one error
        paper_has_error = False
        paper_error_types = set()  # Track all error types for this paper
        
        for event_index, event_data in enumerate(events):
            # Ensure event_data is a dictionary
            if not isinstance(event_data, dict):
                logger.warning(f"Event {event_index} for paper {paper_code} is not a dictionary, skipping.")
                
                if "malformed_events" not in metrics:
                    metrics["malformed_events"] = 0
                metrics["malformed_events"] += 1
                
                # Add to error tracking
                paper_has_error = True
                paper_error_types.add("MALFORMED_EVENT")
                
                # Append original malformed event with error markers
                if isinstance(event_data, str):
                    processed_events.append({
                        "Text": event_data,
                        "Main Action": "ERROR",
                        "Arguments": error_arguments,
                        "Error_Type": "MALFORMED_EVENT_STRING"
                    })
                else:
                    processed_events.append({
                        "Text": str(event_data),
                        "Main Action": "ERROR",
                        "Arguments": error_arguments,
                        "Error_Type": "MALFORMED_EVENT_OBJECT"
                    })
                continue

            event_text = event_data.get("Text", "")
            if not event_text:
                logger.warning(f"Empty event text for paper {paper_code}, event index {event_index}")
                
                if "empty_event_text" not in metrics:
                    metrics["empty_event_text"] = 0
                metrics["empty_event_text"] += 1
                
                # Add to error tracking
                paper_has_error = True
                paper_error_types.add("EMPTY_EVENT_TEXT")
                
                # Add error markers for empty text events
                updated_event = event_data.copy()
                updated_event["Main Action"] = "ERROR"
                updated_event["Arguments"] = error_arguments
                updated_event["Error_Type"] = "EMPTY_EVENT_TEXT"
                processed_events.append(updated_event)
                continue

            # Find the event type (if any specific key exists beyond standard ones)
            event_type_key = None
            standard_keys = {"Text", "Main Action", "Arguments", "Summary"}
            for key in event_data.keys():
                if key not in standard_keys:
                    event_type_key = key
                    break

            # Construct path to the corresponding raw output file
            raw_file_path = os.path.join(raw_output_dir, f"{paper_code}_event_{event_index}.txt")

            if os.path.exists(raw_file_path):
                main_action, arguments, summary, has_error, error_type = process_event(raw_file_path, logger, metrics)

                if has_error:
                    paper_has_error = True
                    paper_error_types.add(error_type)
                
                # Create updated event dictionary, starting with original
                updated_event = event_data.copy()

                # Update with extracted values, using ERROR markers for failures
                updated_event["Main Action"] = main_action if not has_error else "ERROR"
                
                # Use ERROR markers if parsing failure occurred
                if has_error:
                    updated_event["Arguments"] = error_arguments
                    updated_event["Error_Type"] = error_type
                    
                    # Track complete parsing failures in metrics
                    if "complete_parsing_failures" not in metrics:
                        metrics["complete_parsing_failures"] = 0
                    metrics["complete_parsing_failures"] += 1
                else:
                    updated_event["Arguments"] = arguments if arguments else default_arguments

                # Add summary: Use specific event type key if found, otherwise 'Summary'
                if summary and summary != "<NONE>":
                    summary_key = event_type_key if event_type_key else "Summary"
                    updated_event[summary_key] = summary
                    # Remove the original event_type_key if it was just a placeholder text value
                    if event_type_key and event_type_key != "Summary" and event_data.get(event_type_key) == event_text:
                        logger.debug(f"Replacing placeholder key '{event_type_key}' with Summary content.")
                        # Decide if we should remove the old key if summary replaces its content
                        # del updated_event[event_type_key] # Optional: remove original type key if replaced by summary

                processed_events.append(updated_event)
            else:
                logger.warning(f"Raw output file not found for event: {raw_file_path}")
                
                if "missing_event_files" not in metrics:
                    metrics["missing_event_files"] = 0
                metrics["missing_event_files"] += 1
                
                # Add to error tracking
                paper_has_error = True
                paper_error_types.add("MISSING_EVENT_FILE")
                
                # Append original event but mark with ERROR values and error type
                updated_event = event_data.copy()
                updated_event["Main Action"] = "ERROR" 
                updated_event["Arguments"] = error_arguments
                updated_event["Error_Type"] = "MISSING_RAW_FILE"
                processed_events.append(updated_event)
        
        # Update paper-level error metrics
        if paper_has_error:
            if "papers_with_errors" not in metrics:
                metrics["papers_with_errors"] = 0
            metrics["papers_with_errors"] += 1
            
            # Add paper to each error type category it belongs to
            if "papers_by_error_type" not in metrics:
                metrics["papers_by_error_type"] = {}
                
            for error_type in paper_error_types:
                if error_type not in metrics["papers_by_error_type"]:
                    metrics["papers_by_error_type"][error_type] = []
                if paper_code not in metrics["papers_by_error_type"][error_type]:
                    metrics["papers_by_error_type"][error_type].append(paper_code)

    # Return the paper structure with processed events
    return {
        "paper_code": paper_code,
        "abstract": abstract,
        "events": processed_events
    }


def process_files(input_dir: str, raw_output_dir: str, logger: logging.Logger) -> Dict:
    """
    Process all JSON files in the input directory, orchestrating paper processing.

    Args:
        input_dir: Directory containing input JSON files (paper structures)
        raw_output_dir: Directory containing raw LLM output text files
        logger: Logger instance

    Returns:
        Processing metrics dictionary
    """
    # Create output directories relative to the raw_output_dir's parent
    parent_dir = os.path.dirname(raw_output_dir) # Assumes raw_output is inside model/domain folder
    if not parent_dir or parent_dir == raw_output_dir: # Handle edge case if raw_output_dir is top-level
        parent_dir = os.path.dirname(input_dir) # Fallback: use input dir's parent

    annotated_dir = os.path.join(parent_dir, "annotated")
    metrics_dir = os.path.join(parent_dir, "metrics")
    error_dir = os.path.join(parent_dir, "error_files")  # New directory for error files

    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)  # Create error directory
    
    logger.info(f"Annotated files will be saved to: {annotated_dir}")
    logger.info(f"Metrics will be saved to: {metrics_dir}")
    logger.info(f"Error reports will be saved to: {error_dir}")

    # Initialize metrics
    metrics = {
        "input_files_processed": 0,
        "papers_processed": 0,
        "events_processed": 0,  # Total events across all papers
        "successful_extractions": 0,  # Count of successful JSON parses from raw files
        "failed_extractions": 0,  # Count of failed JSON parses from raw files
        "complete_parsing_failures": 0,  # Count of events marked with ERROR
        "papers_with_errors": 0,  # Count of papers that have at least one error
        "missing_raw_files": 0,  # Count of missing raw files for full abstracts
        "missing_event_files": 0,  # Count of missing raw files for events
        "malformed_events": 0,  # Count of events that aren't dictionaries
        "empty_event_text": 0,  # Count of events with empty text
        "failed_files": [],  # List to track specific raw files that failed parsing
        "failed_files_by_type": {},  # Dict to track files by error type
        "papers_by_error_type": {},  # Dict to track papers by error type
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_time_seconds": None
    }

    # Get all JSON files in the input directory
    try:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    except FileNotFoundError:
        logger.error(f"Input directory not found: {input_dir}")
        return metrics # Return empty metrics if input dir missing

    # Process each file
    all_processed_papers = []  # Accumulate papers from all input files
    papers_with_errors = []  # Track papers with errors for separate output

    for json_file in json_files:
        input_file_path = os.path.join(input_dir, json_file)
        logger.info(f"--- Processing input file: {json_file} ---")

        papers_in_file = []  # List to hold papers from the current input file
        error_papers_in_file = []  # List to hold papers with errors from current file

        # Load the input JSON file
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Handle different possible JSON formats in the input file
            current_file_papers = []

            # Try parsing as a JSON array first (most common)
            if file_content.strip().startswith('['):
                try:
                    current_file_papers = json.loads(file_content)
                    logger.info(f"Parsed {len(current_file_papers)} papers from JSON array format in {json_file}")
                except json.JSONDecodeError as e_arr:
                    logger.error(f"Failed to parse {json_file} as JSON array: {e_arr}. Trying other formats.")
                    # Attempt newline-delimited as a fallback even if it starts with [
                    try:
                        current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file} (fallback).")
                    except json.JSONDecodeError as e_ndl_fallback:
                        logger.error(f"Failed newline-delimited fallback for {json_file}: {e_ndl_fallback}. Skipping file.")
                        continue # Skip this file if all parsing fails

            # Try parsing as a single JSON object
            elif file_content.strip().startswith('{'):
                try:
                    paper = json.loads(file_content)
                    current_file_papers = [paper] # Wrap single object in a list
                    logger.info(f"Parsed 1 paper from single JSON object format in {json_file}")
                except json.JSONDecodeError as e_obj:
                    logger.error(f"Failed to parse {json_file} as single JSON object: {e_obj}. Trying newline-delimited.")
                    # Attempt newline-delimited as a fallback
                    try:
                        current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file} (fallback).")
                    except json.JSONDecodeError as e_ndl_fallback2:
                        logger.error(f"Failed newline-delimited fallback for {json_file}: {e_ndl_fallback2}. Skipping file.")
                        continue # Skip file

            # Try parsing as newline-delimited JSON
            else:
                try:
                    current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                    if current_file_papers:
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file}")
                    else:
                        logger.warning(f"No valid JSON objects found in newline-delimited file: {json_file}")
                        # Continue processing other files even if one is empty/invalid
                except json.JSONDecodeError as e_ndl:
                    logger.error(f"Failed to parse {json_file} as newline-delimited JSON: {e_ndl}. Skipping file.")
                    continue # Skip this file

            # Check if any papers were successfully loaded
            if not current_file_papers:
                logger.warning(f"No papers loaded from {json_file}. Skipping file.")
                continue

            # Process each paper found in the current file
            processed_papers_for_this_file = []
            for paper_data in current_file_papers:
                if not isinstance(paper_data, dict):
                    logger.warning(f"Found non-dictionary item in {json_file}, skipping: {type(paper_data)}")
                    continue

                paper_code = paper_data.get("paper_code")
                abstract = paper_data.get("abstract", "")
                events = paper_data.get("events", []) # Default to empty list if missing

                if not paper_code:
                    logger.warning(f"Paper data in {json_file} missing 'paper_code', skipping this paper.")
                    continue

                # Core processing logic for the paper
                processed_paper = process_paper(paper_code, abstract, events, raw_output_dir, logger, metrics)
                processed_papers_for_this_file.append(processed_paper)

                # Check if paper has any errors to track it separately
                paper_has_errors = False
                for event in processed_paper.get("events", []):
                    if event.get("Main Action") == "ERROR" or "Error_Type" in event:
                        paper_has_errors = True
                        break
                
                if paper_has_errors:
                    error_papers_in_file.append(processed_paper)

                # Update metrics
                metrics["papers_processed"] += 1
                metrics["events_processed"] += len(processed_paper.get("events", [])) # Count events in processed output

            # Add processed papers from this file to the overall list
            all_processed_papers.extend(processed_papers_for_this_file)
            papers_with_errors.extend(error_papers_in_file)

            # Save the processed papers *from this input file* to its own annotated file
            output_filename = f"{os.path.splitext(json_file)[0]}_annotated.json"
            output_file_path = os.path.join(annotated_dir, output_filename)
            try:
                with open(output_file_path, "w", encoding="utf-8") as f_out:
                    json.dump({"papers": processed_papers_for_this_file}, f_out, indent=2, ensure_ascii=False)
                logger.info(f"Saved annotations for {json_file} to {output_filename}")
            except Exception as write_err:
                logger.error(f"Failed to write annotated file {output_file_path}: {write_err}")
            
            # Save papers with errors to a separate error file
            if error_papers_in_file:
                error_filename = f"{os.path.splitext(json_file)[0]}_errors.json"
                error_file_path = os.path.join(error_dir, error_filename)
                try:
                    with open(error_file_path, "w", encoding="utf-8") as f_err:
                        json.dump({"papers": error_papers_in_file}, f_err, indent=2, ensure_ascii=False)
                    logger.info(f"Saved error papers for {json_file} to {error_filename}")
                except Exception as write_err:
                    logger.error(f"Failed to write error file {error_file_path}: {write_err}")

            metrics["input_files_processed"] += 1
        except FileNotFoundError:
            logger.error(f"Input JSON file not found: {input_file_path}")
        except Exception as e:
            logger.error(f"Unexpected error processing input file {json_file}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue to the next file

    # Save all processed papers from all files into one aggregated file
    all_papers_file_path = os.path.join(annotated_dir, "all_papers_annotated.json")
    try:
        with open(all_papers_file_path, "w", encoding="utf-8") as f_all:
            json.dump({"papers": all_processed_papers}, f_all, indent=2, ensure_ascii=False)
        logger.info(f"Saved aggregated annotations for all papers to {all_papers_file_path}")
    except Exception as write_err:
        logger.error(f"Failed to write aggregated annotated file {all_papers_file_path}: {write_err}")
    
    # Save all papers with errors into one aggregated error file
    if papers_with_errors:
        all_errors_file_path = os.path.join(error_dir, "all_papers_with_errors.json")
        try:
            with open(all_errors_file_path, "w", encoding="utf-8") as f_err_all:
                json.dump({"papers": papers_with_errors}, f_err_all, indent=2, ensure_ascii=False)
            logger.info(f"Saved aggregated papers with errors to {all_errors_file_path}")
        except Exception as write_err:
            logger.error(f"Failed to write aggregated error file {all_errors_file_path}: {write_err}")

    # Generate detailed error reports by error type
    if "failed_files_by_type" in metrics and metrics["failed_files_by_type"]:
        for error_type, error_files in metrics["failed_files_by_type"].items():
            if error_files:
                error_type_filename = f"error_files_{error_type}.txt"
                error_type_path = os.path.join(error_dir, error_type_filename)
                try:
                    with open(error_type_path, "w", encoding="utf-8") as f_type:
                        f_type.write(f"Files with error type: {error_type}\n")
                        f_type.write(f"Count: {len(error_files)}\n\n")
                        for file_path in sorted(error_files):
                            f_type.write(f"{file_path}\n")
                    logger.info(f"Saved error report for type '{error_type}' with {len(error_files)} files")
                except Exception as write_err:
                    logger.error(f"Failed to write error type report for {error_type}: {write_err}")

    # Generate report of papers by error type
    if "papers_by_error_type" in metrics and metrics["papers_by_error_type"]:
        papers_by_error_path = os.path.join(error_dir, "papers_by_error_type.json")
        try:
            with open(papers_by_error_path, "w", encoding="utf-8") as f_papers:
                json.dump(metrics["papers_by_error_type"], f_papers, indent=2, ensure_ascii=False)
            logger.info(f"Saved papers by error type report")
        except Exception as write_err:
            logger.error(f"Failed to write papers by error type report: {write_err}")

    # Finalize and save metrics
    metrics["end_time"] = datetime.now().isoformat()
    try:
        start = datetime.fromisoformat(metrics["start_time"])
        end = datetime.fromisoformat(metrics["end_time"])
        metrics["total_time_seconds"] = (end - start).total_seconds()
    except Exception as time_err:
        logger.error(f"Error calculating total processing time: {time_err}")

    # Save detailed metrics
    metrics_file_path = os.path.join(metrics_dir, "processing_metrics.json")
    try:
        with open(metrics_file_path, "w", encoding="utf-8") as f_metrics:
            json.dump(metrics, f_metrics, indent=2)
        logger.info(f"Processing metrics saved to {metrics_file_path}")
    except Exception as write_err:
        logger.error(f"Failed to write metrics file {metrics_file_path}: {write_err}")
    
    # Save a human-readable metrics summary
    metrics_summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
    try:
        with open(metrics_summary_path, "w", encoding="utf-8") as f_summary:
            f_summary.write("=== Argument Extraction Processing Summary ===\n\n")
            f_summary.write(f"Start time: {metrics['start_time']}\n")
            f_summary.write(f"End time: {metrics['end_time']}\n")
            f_summary.write(f"Total processing time: {metrics.get('total_time_seconds', 'N/A'):.2f} seconds\n\n")
            
            f_summary.write("=== Overall Statistics ===\n")
            f_summary.write(f"Input files processed: {metrics['input_files_processed']}\n")
            f_summary.write(f"Papers processed: {metrics['papers_processed']}\n")
            f_summary.write(f"Total events processed: {metrics['events_processed']}\n")
            f_summary.write(f"Papers with at least one error: {metrics.get('papers_with_errors', 0)}\n\n")
            
            f_summary.write("=== Success/Failure Statistics ===\n")
            f_summary.write(f"Successful raw file extractions: {metrics['successful_extractions']}\n")
            f_summary.write(f"Failed raw file extractions: {metrics['failed_extractions']}\n")
            f_summary.write(f"Complete parsing failures (ERRORs): {metrics.get('complete_parsing_failures', 0)}\n")
            f_summary.write(f"Missing raw files (full abstract): {metrics.get('missing_raw_files', 0)}\n")
            f_summary.write(f"Missing event files: {metrics.get('missing_event_files', 0)}\n")
            f_summary.write(f"Malformed events: {metrics.get('malformed_events', 0)}\n")
            f_summary.write(f"Empty event text: {metrics.get('empty_event_text', 0)}\n\n")
            
            # Success details by type
            f_summary.write("=== Success Details by Type ===\n")
            for key in sorted([k for k in metrics.keys() if k.startswith("success_type_")]):
                success_type = key.replace("success_type_", "")
                count = metrics[key]
                f_summary.write(f"{success_type}: {count}\n")
            f_summary.write("\n")
            
            # Error details by type
            f_summary.write("=== Error Details by Type ===\n")
            for key in sorted([k for k in metrics.keys() if k.startswith("error_type_")]):
                error_type = key.replace("error_type_", "")
                count = metrics[key]
                f_summary.write(f"{error_type}: {count}\n")
            f_summary.write("\n")
            
            # Failed files summary
            if metrics["failed_files"]:
                f_summary.write(f"=== Raw Files with Failed Parsing ({len(metrics['failed_files'])}) ===\n")
                f_summary.write("See error_files_*.txt for detailed lists by error type\n\n")
            else:
                f_summary.write("=== No raw files failed JSON parsing ===\n\n")
                
        logger.info(f"Human-readable metrics summary saved to {metrics_summary_path}")
    except Exception as write_err:
        logger.error(f"Failed to write metrics summary file {metrics_summary_path}: {write_err}")

    # Log summary results
    logger.info("--- Processing Summary ---")
    logger.info(f"Input files processed: {metrics['input_files_processed']}")
    logger.info(f"Papers processed: {metrics['papers_processed']}")
    logger.info(f"Total events processed: {metrics['events_processed']}")
    logger.info(f"Papers with at least one error: {metrics.get('papers_with_errors', 0)}")
    logger.info(f"Successful raw file extractions: {metrics['successful_extractions']}")
    logger.info(f"Failed raw file extractions: {metrics['failed_extractions']}")
    logger.info(f"Complete parsing failures (ERRORs): {metrics.get('complete_parsing_failures', 0)}")
    logger.info(f"Total processing time: {metrics['total_time_seconds']:.2f} seconds" if isinstance(metrics.get('total_time_seconds'), float) else "N/A")

    # Report specific failed files at the end for clarity
    if metrics["failed_files"]:
        logger.info(f"--- Raw files that failed JSON parsing ({len(metrics['failed_files'])}) ---")
        # Sort for consistency, limit logged files if list is very long
        sorted_failed_files = sorted(list(set(metrics['failed_files']))) # Use set to remove duplicates
        max_log_files = 50
        for i, failed_file in enumerate(sorted_failed_files):
            if i < max_log_files:
                logger.info(f"  - {failed_file}")
            elif i == max_log_files:
                logger.info(f"  - ... (and {len(sorted_failed_files) - max_log_files} more)")
                break
    else:
        logger.info("--- No raw files failed JSON parsing ---")

    return metrics


def main():
    """Main function to parse arguments and run the argument extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract argument structures from raw LLM output files, cleaning common JSON errors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
        )
    parser.add_argument("--domains", type=str, required=True, nargs='+',
                        help="List of domains to process (e.g., ACL BIOINFO). Used to find input/output subdirectories.")
    parser.add_argument("--base-dir", type=str, default="./output/given/event_extraction",
                        help="Base directory where prompt/model/domain output structures reside.")
    parser.add_argument("--input-dir", type=str, default="./data/input",
                        help="Base directory containing domain-specific input JSON files (e.g., ./SciEvent_data/raw/ACL).")
    parser.add_argument("--prompt-template", type=str, default="Oneshot",
                        help="Name of the prompt template used (subdirectory name under base-dir).")
    parser.add_argument("--model-name", type=str, default="gpt-4.1", #Qwen2.5-7B-Instruct Meta-Llama-3-8B-Instruct DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B
                        help="Name of the model used (subdirectory name under prompt-template).")

    args = parser.parse_args()

    # Process each domain one by one
    for domain in args.domains:
        # Construct specific paths based on arguments
        # Input path for the specific domain
        domain_input_dir = os.path.join(args.input_dir, domain)
        # Base output path for this specific experiment run
        output_base_dir = os.path.join(args.base_dir, args.prompt_template, args.model_name, domain)
        # Specific subdirectories within the output base
        raw_output_dir = os.path.join(output_base_dir, "raw_output")
        log_dir = os.path.join(output_base_dir, "logs") # Logs stored within the specific run directory

        # Set up logging (logs will go into the specific output_base_dir/logs)
        logger = setup_logging(log_dir)

        logger.info("=================================================")
        logger.info(f"Starting Argument Extraction Process")
        logger.info(f"Domain: {domain}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Prompt Template: {args.prompt_template}")
        logger.info(f"Base Output Dir: {args.base_dir}")
        logger.info(f"Input Data Dir (Domain Specific): {domain_input_dir}")
        logger.info(f"Raw LLM Output Dir: {raw_output_dir}")
        logger.info(f"Log Directory: {log_dir}")
        logger.info("=================================================")

        # Check if essential directories exist before starting
        if not os.path.exists(domain_input_dir):
            logger.error(f"Input directory not found: {domain_input_dir}")
            logger.error("Please ensure the input data exists.")
            continue # Skip to next domain if input data is missing

        if not os.path.exists(raw_output_dir):
            logger.error(f"Raw output directory not found: {raw_output_dir}")
            logger.error("Please ensure the raw LLM output files are present.")
            continue # Skip to next domain if raw outputs are missing

        # === Start Processing ===
        metrics = process_files(domain_input_dir, raw_output_dir, logger)
        # === End Processing ===

        logger.info("=================================================")
        logger.info(f"Argument Extraction Process Finished.")
        # Log key metrics again at the end
        logger.info(f"Final Metrics Summary:")
        logger.info(f"  Input Files Processed: {metrics.get('input_files_processed', 'N/A')}")
        logger.info(f"  Papers Processed: {metrics.get('papers_processed', 'N/A')}")
        logger.info(f"  Successful Extractions: {metrics.get('successful_extractions', 'N/A')}")
        logger.info(f"  Failed Extractions: {metrics.get('failed_extractions', 'N/A')}")
        logger.info(f"  Complete Parsing Failures (ERRORs): {metrics.get('complete_parsing_failures', 'N/A')}")
        logger.info(f"  Total Time: {metrics.get('total_time_seconds', 'N/A'):.2f} seconds" if isinstance(metrics.get('total_time_seconds'), float) else "N/A")
        logger.info("=================================================")


if __name__ == "__main__":
    main()