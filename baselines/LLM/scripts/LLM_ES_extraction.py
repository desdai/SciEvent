#!/usr/bin/env python3
"""
LLM Abstract Chunker - Processes raw LLM output files to extract chunked sections from abstracts.
Outputs sections in direct bracket notation format: [Background], [Method], [Results], [Implications]

Features:
1. Handles both raw LLM outputs and already-chunked abstracts
2. Processes multiple variations of END_OF_CHUNK markers 
3. Properly handles various formats of NONE markers (<None>, <NONE>, none, None)
4. Removes all newlines from abstract sections
5. Handles multiple instances of the same section headers
6. Prevents content duplication in the final output
7. Improved pattern matching for diverse section header formats
8. Better handling of multiline section content
9. Prevents overwriting existing content with <NONE> markers
10. Supports processing multiple domains in one go
11. Enhanced debugging to track section processing
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"chunking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("abstract_chunker")


def is_none_content(content: str) -> bool:
    """
    Check if content represents a NONE value in any format.
    
    Args:
        content: Content to check
    
    Returns:
        True if content represents NONE, False otherwise
    """
    # Clean up the content
    cleaned = content.strip()
    
    # Check for empty string
    if not cleaned:
        return True
    
    # Check for simple placeholders
    if cleaned in ["...", "```", '"""']:
        return True
    
    # Check various NONE formats (case-insensitive)
    patterns = [
        r'^<none>$',        # <None>, <NONE>, <none>
        r'^none\.?$',       # none, None, none.
        r'^\s*<\s*none\s*>\s*$',  # < None >, < none >, etc.
        r'^\.\.\.+$',       # ..., ...., etc.
        r'^`{3,}$',         # ```, ````, etc.
        r'^"{3,}$',         # """, """", etc.
    ]
    
    for pattern in patterns:
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            return True
    
    return False


def extract_chunked_sections(content: str, logger: logging.Logger, debug_file_path: Optional[str] = None) -> Tuple[Dict[str, str], bool]:
    """
    Extract sections from content that might be in chunked format.
    
    Args:
        content: Content to parse
        logger: Logger instance
        debug_file_path: Optional path to write detailed debug info
    
    Returns:
        Tuple containing:
        - Dictionary with extracted sections
        - Boolean indicating if content was already chunked
    """
    # Default sections
    sections = {
        "[Background]": "<NONE>",
        "[Method]": "<NONE>",
        "[Results]": "<NONE>",
        "[Implications]": "<NONE>"
    }
    
    # Create debug file if path provided
    if debug_file_path:
        debug_dir = os.path.dirname(debug_file_path)
        os.makedirs(debug_dir, exist_ok=True)
        
        with open(debug_file_path, "a", encoding="utf-8") as f_debug:
            f_debug.write("\n=== EXTRACT_CHUNKED_SECTIONS FUNCTION ===\n")
            f_debug.write(f"Analyzing content of length: {len(content)} characters\n")
            f_debug.write("First 100 chars: " + content[:100] + "...\n")
            f_debug.write("Last 100 chars: ..." + content[-100:] + "\n\n")
    
    # First try the direct approach: split by section headers
    found_chunked = False
    section_markers = ["[Background]", "[Method]", "[Results]", "[Implications]"]
    sections_found = []
    
    for marker in section_markers:
        if marker in content:
            sections_found.append(marker)
            found_chunked = True
            
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"Found section marker: {marker} at position {content.find(marker)}\n")
    
    # If we found section markers, process them directly
    if sections_found:
        if debug_file_path:
            with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                f_debug.write(f"\nFound {len(sections_found)} section markers using direct approach\n")
                f_debug.write(f"Sections found (in order): {sections_found}\n\n")
        
        # Sort by their position in the text
        sections_found.sort(key=lambda s: content.find(s))
        
        # Process each section
        for i, section in enumerate(sections_found):
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"Processing section {i+1}/{len(sections_found)}: {section}\n")
            
            # Find the start of this section's content (after the section header)
            start_pos = content.find(section) + len(section)
            
            # Skip any whitespace and colon
            orig_start_pos = start_pos
            while start_pos < len(content) and (content[start_pos].isspace() or content[start_pos] == ':'):
                start_pos += 1
            
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"  Section header ends at position: {orig_start_pos}\n")
                    f_debug.write(f"  Content starts at position: {start_pos}\n")
                    if start_pos < len(content):
                        f_debug.write(f"  First 10 chars of content: '{content[start_pos:start_pos+10]}'\n")
            
            # Find the end (next section or end of content)
            if i < len(sections_found) - 1:
                end_pos = content.find(sections_found[i+1])
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  Not the last section. Next section '{sections_found[i+1]}' starts at position {end_pos}\n")
            else:
                # For the last section, check for END_OF_CHUNK markers
                end_pos = len(content)
                
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  This is the last section. Searching for end markers after position {start_pos}\n")
                        if section == "[Implications]":
                            f_debug.write(f"  This is the Implications section - also checking for backtick markers\n")
                
                # Define end markers - standard ones plus special ones for Implications
                end_markers = ["<|END_OF_CHUNK|>", "<|END_OF CHUNK|>", "<|ENDOFCHUNK|>", "<|END OF CHUNK|>", "<|END_OF Chunk|>"]
                
                # Add backtick marker specifically for the Implications section
                if section == "[Implications]":
                    end_markers.append("```")
                
                # Check for all end markers
                for marker in end_markers:
                    marker_pos = content.find(marker, start_pos)
                    if marker_pos != -1 and marker_pos < end_pos:
                        end_pos = marker_pos
                        if debug_file_path:
                            with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                                f_debug.write(f"  Found end marker '{marker}' at position {marker_pos}\n")
                                f_debug.write(f"  Setting end position to {end_pos}\n")
                                content_preview = content[start_pos:end_pos][:100] + ("..." if len(content[start_pos:end_pos]) > 100 else "")
                                f_debug.write(f"  Content before marker: '{content_preview}'\n")
            
            # Extract and clean the section content
            section_content = content[start_pos:end_pos].strip()
            
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"  Raw section content length: {len(section_content)} characters\n")
                    if len(section_content) > 0:
                        preview = section_content[:100] + ("..." if len(section_content) > 100 else "")
                        f_debug.write(f"  Raw section content sample: '{preview}'\n")
                    else:
                        f_debug.write("  Raw section content is empty\n")
            
            # Special handling for Implications section with placeholder content
            if section == "[Implications]" and section_content in ["...", "```", "\"\"\"", "'''"]:
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  Implications section contains only '{section_content}', treating as <NONE>\n")
                
                sections[section] = "<NONE>"
                continue
            
            # Check if it's NONE content using the standard function
            is_none = is_none_content(section_content)
            
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"  Is section content marked as NONE? {is_none}\n")
            
            if is_none:
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  Content identified as NONE. Setting to <NONE>\n")
                
                sections[section] = "<NONE>"
                continue
            
            # Clean content - remove newlines and normalize whitespace
            section_content = re.sub(r'\n', ' ', section_content)
            section_content = re.sub(r'\s+', ' ', section_content).strip()
            
            # Check again after cleaning if the Implications section just has placeholders
            if section == "[Implications]" and section_content in ["...", "```", "\"\"\"", "'''"]:
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  After cleaning, Implications section contains only '{section_content}', treating as <NONE>\n")
                
                sections[section] = "<NONE>"
                continue
                
            if debug_file_path:
                with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"  Cleaned section content length: {len(section_content)} characters\n")
                    preview = section_content[:100] + ("..." if len(section_content) > 100 else "")
                    f_debug.write(f"  Cleaned section content: '{preview}'\n")
            
            # Store in our sections dictionary
            if sections[section] == "<NONE>":
                sections[section] = section_content
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  Saved to {section}\n")
                logger.info(f"Extracted {section}: {section_content[:50]}...")
            else:
                # Append if there's already content
                sections[section] += " " + section_content
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                        f_debug.write(f"  Appended to existing {section} content\n")
                logger.info(f"Appended {section} content: {section_content[:50]}...")
        
        # Log the final extracted sections
        if debug_file_path:
            with open(debug_file_path, "a", encoding="utf-8") as f_debug:
                f_debug.write("\nFinal extracted sections:\n")
                for section_name, section_content in sections.items():
                    if section_content != "<NONE>":
                        preview = section_content[:50] + ("..." if len(section_content) > 50 else "")
                        f_debug.write(f"{section_name}: {preview}\n")
                    else:
                        f_debug.write(f"{section_name}: <NONE>\n")
        
        return sections, found_chunked
    
    # If direct approach failed, alternative approaches could go here
    # For simplicity, we'll just log this case and return not chunked
    if debug_file_path:
        with open(debug_file_path, "a", encoding="utf-8") as f_debug:
            f_debug.write("Direct approach didn't find sections. Returning not chunked.\n")
    
    return sections, found_chunked


def extract_sections_from_raw_output(raw_output_path: str, logger: logging.Logger) -> Tuple[Dict[str, str], str]:
    """
    Extract the four abstract sections from raw LLM output files.
    
    Args:
        raw_output_path: Path to the raw output file
        logger: Logger instance
    
    Returns:
        Tuple containing:
        - Dictionary with the four sections using bracket notation
        - Error type string describing the failure reason if any
    """
    # Default empty sections
    sections = {
        "[Background]": "<NONE>",
        "[Method]": "<NONE>",
        "[Results]": "<NONE>",
        "[Implications]": "<NONE>"
    }
    
    # Define path for enhanced debug file
    debug_dir = os.path.join(os.path.dirname(raw_output_path), "enhanced_debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = os.path.join(debug_dir, f"{os.path.basename(raw_output_path)}_enhanced.log")
    
    # Start fresh debug file
    with open(debug_file, "w", encoding="utf-8") as f_debug:
        f_debug.write(f"=== ENHANCED DEBUGGING FOR {os.path.basename(raw_output_path)} ===\n\n")
        f_debug.write("STEP 1: Reading raw output file\n")
    
    try:
        # Read the raw output file
        with open(raw_output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Log the raw output for debugging
        with open(debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write(f"Raw file length: {len(content)} characters\n")
            f_debug.write(f"Raw file content:\n{content}\n\n")
            f_debug.write("STEP 2: Checking if content is already chunked\n")
        
        # First check if the content is already in properly chunked format
        chunked_sections, is_chunked = extract_chunked_sections(content, logger, debug_file)
        
        if is_chunked:
            with open(debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write("Content is already in chunked format. Using extracted sections.\n")
                f_debug.write("Sections extracted from chunked content:\n")
                for section, text in chunked_sections.items():
                    f_debug.write(f"  {section}: {text[:100]}...\n" if text != "<NONE>" else f"  {section}: <NONE>\n")
            
            return chunked_sections, "SUCCESS"
        
        # If we get here, the content wasn't already in chunked format
        with open(debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("\nSTEP 3: Looking for END_OF_CHUNK markers\n")
        
        # Look specifically for END_OF_CHUNK markers
        original_content = content
        end_of_chunk_markers = ["<|END_OF_CHUNK|>", "<|END_OF CHUNK|>", "<|ENDOFCHUNK|>", "<|END OF CHUNK|>", "<|END_OF Chunk|>"]
        
        for marker in end_of_chunk_markers:
            if marker in content:
                # Only process content up to the first occurrence of END_OF_CHUNK
                marker_pos = content.find(marker)
                truncated_content = content[:marker_pos].strip()
                
                with open(debug_file, "a", encoding="utf-8") as f_debug:
                    f_debug.write(f"Found marker '{marker}' at position {marker_pos}\n")
                    f_debug.write(f"Truncating content to everything before this marker.\n")
                    f_debug.write(f"Truncated content length: {len(truncated_content)} characters\n")
                    f_debug.write(f"Last 100 chars of truncated content: ...{truncated_content[-100:]}\n\n")
                
                content = truncated_content
                break
        
        # Check if we found and processed an end marker
        if content != original_content:
            with open(debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write("Content was truncated at an end marker.\n")
        else:
            with open(debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write("No end markers found. Using full content.\n")
        
        # Try again to extract sections after processing end markers
        with open(debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("\nSTEP 4: Re-checking if processed content is in chunked format\n")
        
        chunked_sections, is_chunked = extract_chunked_sections(content, logger, debug_file)
        
        if is_chunked:
            with open(debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write("Processed content is in chunked format. Using extracted sections.\n")
                f_debug.write("Final sections:\n")
                for section, text in chunked_sections.items():
                    f_debug.write(f"  {section}: {text[:100]}...\n" if text != "<NONE>" else f"  {section}: <NONE>\n")
            
            # Special check for Implications section
            if chunked_sections["[Implications]"] == "<NONE>":
                with open(debug_file, "a", encoding="utf-8") as f_debug:
                    f_debug.write("\nWARNING: Implications section is <NONE> - special debugging:\n")
                    
                    # Check if "[Implications]" exists in the content
                    impl_pos = content.find("[Implications]")
                    if impl_pos != -1:
                        f_debug.write(f"[Implications] marker found at position {impl_pos}\n")
                        
                        # Find the beginning of content after marker
                        start_pos = impl_pos + len("[Implications]")
                        while start_pos < len(content) and (content[start_pos].isspace() or content[start_pos] == ':'):
                            start_pos += 1
                        
                        f_debug.write(f"Content after [Implications] marker starts at position {start_pos}\n")
                        
                        # Get the raw content
                        end_pos = len(content)
                        for marker in end_of_chunk_markers:
                            marker_pos = content.find(marker, start_pos)
                            if marker_pos != -1 and marker_pos < end_pos:
                                end_pos = marker_pos
                                f_debug.write(f"End marker '{marker}' found at position {marker_pos}\n")
                        
                        raw_impl_content = content[start_pos:end_pos].strip()
                        f_debug.write(f"Raw Implications content length: {len(raw_impl_content)} characters\n")
                        f_debug.write(f"Raw Implications content: '{raw_impl_content}'\n")
                        
                        # Check if it's being categorized as NONE
                        is_none = is_none_content(raw_impl_content)
                        f_debug.write(f"Is raw content identified as NONE? {is_none}\n")
                        
                        # Try manual extraction as a last resort
                        if not is_none and raw_impl_content:
                            f_debug.write("Attempting manual extraction of Implications section\n")
                            clean_impl_content = re.sub(r'\n', ' ', raw_impl_content)
                            clean_impl_content = re.sub(r'\s+', ' ', clean_impl_content).strip()
                            
                            # Override the NONE value with our manual extraction
                            chunked_sections["[Implications]"] = clean_impl_content
                            f_debug.write(f"Manually extracted Implications: {clean_impl_content[:100]}...\n")
                    else:
                        f_debug.write("[Implications] marker not found in content\n")
            
            return chunked_sections, "SUCCESS"
        
        # We've tried everything and failed to extract sections
        with open(debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("\nSTEP 5: All extraction attempts failed. Returning default sections.\n")
            for section, text in sections.items():
                f_debug.write(f"  {section}: {text}\n")
        
        return sections, "NO_SECTIONS_FOUND"
    
    except Exception as e:
        logger.error(f"Unexpected error processing {raw_output_path}: {e}")
        with open(debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write(f"\nERROR: Exception occurred: {str(e)}\n")
            import traceback
            f_debug.write(f"Traceback: {traceback.format_exc()}\n")
        
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return sections, "UNEXPECTED_ERROR"


def process_paper(paper_code: str, abstract: str, raw_output_dir: str, logger: logging.Logger, metrics: Dict) -> Dict:
    """
    Process a single paper, extracting chunked sections from its raw output file.
    
    Args:
        paper_code: The paper code (used for finding raw files)
        abstract: The paper's abstract text
        raw_output_dir: Directory containing raw LLM output files
        logger: Logger instance
        metrics: Metrics dictionary to update
    
    Returns:
        Processed paper dictionary with chunked sections in bracket notation format
    """
    logger.info(f"Processing paper: {paper_code}")
    
    # Setup enhanced debugging
    debug_dir = os.path.join(raw_output_dir, "enhanced_debug")
    os.makedirs(debug_dir, exist_ok=True)
    paper_debug_file = os.path.join(debug_dir, f"{paper_code}_processing.log")
    
    with open(paper_debug_file, "w", encoding="utf-8") as f_debug:
        f_debug.write(f"=== PAPER PROCESSING: {paper_code} ===\n\n")
    
    # Initialize the paper structure with default values
    paper_data = {
        "abstract": abstract,
        "[Background]": "<NONE>",
        "[Method]": "<NONE>",
        "[Results]": "<NONE>",
        "[Implications]": "<NONE>"
    }
    
    with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
        f_debug.write("STEP 1: Initialized paper data structure\n")
        f_debug.write("STEP 2: Checking if abstract is already chunked\n")
    
    # First, check if the abstract itself is already chunked
    chunked_sections, is_chunked = extract_chunked_sections(abstract, logger, paper_debug_file)
    
    if is_chunked:
        with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("\nAbstract is already chunked. Using section content from abstract.\n")
        
        logger.info(f"Paper {paper_code} has a pre-chunked abstract. Using it directly.")
        for section_key, section_text in chunked_sections.items():
            with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
                f_debug.write(f"Setting {section_key} to: {section_text[:100]}...\n" if section_text != "<NONE>" else f"Setting {section_key} to: <NONE>\n")
            
            paper_data[section_key] = section_text
            if section_text != "<NONE>":
                metrics["sections_extracted"] += 1
        
        metrics["successful_extractions"] += 1
        return paper_data
    
    # Path to the raw output file for this paper
    raw_file_path = os.path.join(raw_output_dir, f"{paper_code}_chunks.txt")
    
    with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
        f_debug.write(f"\nSTEP 3: Abstract not pre-chunked. Looking for raw output file: {raw_file_path}\n")
    
    if os.path.exists(raw_file_path):
        with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("Raw output file found. Extracting sections from raw output.\n")
        
        # Extract sections from the raw output
        sections, error_type = extract_sections_from_raw_output(raw_file_path, logger)
        
        with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write(f"\nSTEP 4: Section extraction completed with status: {error_type}\n")
            f_debug.write("Extracted sections:\n")
            for section, text in sections.items():
                f_debug.write(f"  {section}: {text[:100]}...\n" if text != "<NONE>" else f"  {section}: <NONE>\n")
        
        # Track extraction metrics
        if error_type == "SUCCESS":
            metrics["successful_extractions"] += 1
        else:
            metrics["failed_extractions"] += 1
            
            # Track error types
            error_type_key = f"error_type_{error_type}"
            if error_type_key not in metrics:
                metrics[error_type_key] = 0
            metrics[error_type_key] += 1
            
            # Track failed files
            if "failed_files" not in metrics:
                metrics["failed_files"] = []
            metrics["failed_files"].append(raw_file_path)
        
        # Update paper data with extracted sections
        with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write("\nSTEP 5: Updating paper data with extracted sections\n")
        
        for section_key, section_text in sections.items():
            with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
                before_value = paper_data[section_key]
                f_debug.write(f"  {section_key}: Changing from '{before_value}' to '{section_text[:100]}...'\n" if section_text != "<NONE>" else f"  {section_key}: Keeping as '<NONE>'\n")
            
            paper_data[section_key] = section_text
            if section_text != "<NONE>":
                metrics["sections_extracted"] += 1
    else:
        with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
            f_debug.write(f"Raw output file not found: {raw_file_path}\n")
            f_debug.write("Keeping default sections (all <NONE>)\n")
        
        logger.warning(f"Raw output file not found for paper: {raw_file_path}")
        metrics["missing_raw_files"] += 1
    
    # Log final paper data
    with open(paper_debug_file, "a", encoding="utf-8") as f_debug:
        f_debug.write("\nFINAL PAPER DATA:\n")
        for key, value in paper_data.items():
            if key == "abstract":
                f_debug.write(f"  {key}: [abstract text not shown]\n")
            else:
                f_debug.write(f"  {key}: {value[:100]}...\n" if value != "<NONE>" else f"  {key}: <NONE>\n")
    
    return paper_data


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
    parent_dir = os.path.dirname(raw_output_dir)
    chunked_dir = os.path.join(parent_dir, "chunked")
    metrics_dir = os.path.join(parent_dir, "metrics")
    
    os.makedirs(chunked_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logger.info(f"Chunked files will be saved to: {chunked_dir}")
    logger.info(f"Metrics will be saved to: {metrics_dir}")
    
    # Initialize metrics
    metrics = {
        "input_files_processed": 0,
        "papers_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "sections_extracted": 0,
        "missing_raw_files": 0,
        "failed_files": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_time_seconds": None
    }
    
    # Create master debug log
    debug_dir = os.path.join(raw_output_dir, "enhanced_debug")
    os.makedirs(debug_dir, exist_ok=True)
    master_debug_log = os.path.join(debug_dir, "master_processing_log.txt")
    
    with open(master_debug_log, "w", encoding="utf-8") as master_log:
        master_log.write("=== ABSTRACT CHUNKING MASTER PROCESSING LOG ===\n\n")
        master_log.write(f"Start time: {metrics['start_time']}\n")
        master_log.write(f"Input directory: {input_dir}\n")
        master_log.write(f"Raw output directory: {raw_output_dir}\n\n")
    
    # Get all JSON files in the input directory
    try:
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"Found {len(json_files)} JSON files to process\n")
    except FileNotFoundError:
        logger.error(f"Input directory not found: {input_dir}")
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"ERROR: Input directory not found: {input_dir}\n")
        return metrics
    
    # Process each file
    all_processed_papers = {}
    
    for json_file in json_files:
        input_file_path = os.path.join(input_dir, json_file)
        logger.info(f"--- Processing input file: {json_file} ---")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"\n=== Processing file: {json_file} ===\n")
        
        # Load the input JSON file
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            # Handle different possible JSON formats in the input file
            current_file_papers = []
            
            with open(master_debug_log, "a", encoding="utf-8") as master_log:
                master_log.write(f"Loading file: {input_file_path}\n")
            
            # Try parsing as a JSON array first (most common)
            if file_content.strip().startswith('['):
                try:
                    current_file_papers = json.loads(file_content)
                    logger.info(f"Parsed {len(current_file_papers)} papers from JSON array format in {json_file}")
                    
                    with open(master_debug_log, "a", encoding="utf-8") as master_log:
                        master_log.write(f"Successfully parsed as JSON array with {len(current_file_papers)} papers\n")
                except json.JSONDecodeError:
                    # Try newline-delimited as a fallback
                    try:
                        current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file}")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"Parsed as newline-delimited JSON with {len(current_file_papers)} papers\n")
                    except json.JSONDecodeError as e_ndl_fallback:
                        logger.error(f"Failed newline-delimited fallback for {json_file}: {e_ndl_fallback}. Skipping file.")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"ERROR: Failed to parse file. {str(e_ndl_fallback)}\n")
                        continue
            
            # Try parsing as a single JSON object
            elif file_content.strip().startswith('{'):
                try:
                    file_data = json.loads(file_content)
                    # Check if it's a dictionary of papers or a single paper
                    if any(isinstance(value, dict) and "abstract" in value for value in file_data.values()):
                        # It's a dictionary of papers
                        current_file_papers = [{"paper_code": key, **value} for key, value in file_data.items()]
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"Parsed as dictionary of papers with {len(current_file_papers)} papers\n")
                    else:
                        # It's a single paper
                        current_file_papers = [file_data]
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"Parsed as single paper object\n")
                    
                    logger.info(f"Parsed {len(current_file_papers)} papers from JSON object format in {json_file}")
                except json.JSONDecodeError:
                    # Try newline-delimited as a fallback
                    try:
                        current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file}")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"Parsed as newline-delimited JSON with {len(current_file_papers)} papers\n")
                    except json.JSONDecodeError as e_ndl_fallback2:
                        logger.error(f"Failed newline-delimited fallback for {json_file}: {e_ndl_fallback2}. Skipping file.")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"ERROR: Failed to parse file. {str(e_ndl_fallback2)}\n")
                        continue
            
            # Try parsing as newline-delimited JSON
            else:
                try:
                    current_file_papers = [json.loads(line) for line in file_content.strip().split('\n') if line.strip()]
                    if current_file_papers:
                        logger.info(f"Parsed {len(current_file_papers)} papers from newline-delimited format in {json_file}")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"Parsed as newline-delimited JSON with {len(current_file_papers)} papers\n")
                    else:
                        logger.warning(f"No valid JSON objects found in newline-delimited file: {json_file}")
                        
                        with open(master_debug_log, "a", encoding="utf-8") as master_log:
                            master_log.write(f"WARNING: No valid JSON objects found in file\n")
                except json.JSONDecodeError as e_ndl:
                    logger.error(f"Failed to parse {json_file} as newline-delimited JSON: {e_ndl}. Skipping file.")
                    
                    with open(master_debug_log, "a", encoding="utf-8") as master_log:
                        master_log.write(f"ERROR: Failed to parse file. {str(e_ndl)}\n")
                    continue
            
            # Check if any papers were successfully loaded
            if not current_file_papers:
                logger.warning(f"No papers loaded from {json_file}. Skipping file.")
                
                with open(master_debug_log, "a", encoding="utf-8") as master_log:
                    master_log.write(f"WARNING: No papers loaded from file. Skipping.\n")
                continue
            
            # Process each paper found in the current file
            processed_papers_for_this_file = {}
            
            with open(master_debug_log, "a", encoding="utf-8") as master_log:
                master_log.write(f"Processing {len(current_file_papers)} papers from this file\n")
            
            for paper_data in current_file_papers:
                if not isinstance(paper_data, dict):
                    logger.warning(f"Found non-dictionary item in {json_file}, skipping: {type(paper_data)}")
                    
                    with open(master_debug_log, "a", encoding="utf-8") as master_log:
                        master_log.write(f"WARNING: Skipping non-dictionary item: {type(paper_data)}\n")
                    continue
                
                paper_code = paper_data.get("paper_code")
                abstract = paper_data.get("abstract", "")
                
                if not paper_code:
                    logger.warning(f"Paper data in {json_file} missing 'paper_code', skipping this paper.")
                    
                    with open(master_debug_log, "a", encoding="utf-8") as master_log:
                        master_log.write(f"WARNING: Skipping paper with missing paper_code\n")
                    continue
                
                with open(master_debug_log, "a", encoding="utf-8") as master_log:
                    master_log.write(f"Processing paper: {paper_code}\n")
                
                # Process the paper
                processed_paper = process_paper(paper_code, abstract, raw_output_dir, logger, metrics)
                processed_papers_for_this_file[paper_code] = processed_paper
                all_processed_papers[paper_code] = processed_paper
                
                # Special debug check for Implications section
                if processed_paper["[Implications]"] == "<NONE>":
                    with open(master_debug_log, "a", encoding="utf-8") as master_log:
                        master_log.write(f"NOTE: Paper {paper_code} has <NONE> for [Implications] section\n")
                        master_log.write("Checking if implications should have been extracted...\n")
                        
                        raw_file_path = os.path.join(raw_output_dir, f"{paper_code}_chunks.txt")
                        if os.path.exists(raw_file_path):
                            with open(raw_file_path, "r", encoding="utf-8") as raw_f:
                                raw_content = raw_f.read()
                                
                            if "[Implications]" in raw_content:
                                master_log.write("WARNING: [Implications] marker exists in the raw file but wasn't extracted!\n")
                                
                                # Find the implications section start
                                impl_pos = raw_content.find("[Implications]")
                                start_pos = impl_pos + len("[Implications]")
                                while start_pos < len(raw_content) and (raw_content[start_pos].isspace() or raw_content[start_pos] == ':'):
                                    start_pos += 1
                                
                                # Find end marker
                                end_pos = len(raw_content)
                                end_markers = ["<|END_OF_CHUNK|>", "<|END_OF CHUNK|>", "<|ENDOFCHUNK|>", "<|END OF CHUNK|>", "<|END_OF Chunk|>"]
                                for marker in end_markers:
                                    marker_pos = raw_content.find(marker, start_pos)
                                    if marker_pos != -1 and marker_pos < end_pos:
                                        end_pos = marker_pos
                                        master_log.write(f"Found end marker '{marker}' at position {marker_pos}\n")
                                
                                raw_impl = raw_content[start_pos:end_pos].strip()
                                master_log.write(f"Raw Implications content: '{raw_impl}'\n")
                                
                                # Check if it's being categorized as NONE
                                is_none = is_none_content(raw_impl)
                                master_log.write(f"Is raw content identified as NONE? {is_none}\n")
                        else:
                            master_log.write(f"Raw file not found at {raw_file_path}\n")
                
                # Update metrics
                metrics["papers_processed"] += 1
            
            # Save the processed papers from this input file to its own chunked file
            output_filename = f"{os.path.splitext(json_file)[0]}_chunked.txt"
            output_file_path = os.path.join(chunked_dir, output_filename)
            
            with open(master_debug_log, "a", encoding="utf-8") as master_log:
                master_log.write(f"Saving chunked output to: {output_file_path}\n")
            
            try:
                with open(output_file_path, "w", encoding="utf-8") as f_out:
                    json.dump(processed_papers_for_this_file, f_out, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved chunked sections for {json_file} to {output_filename}")
                
                with open(master_debug_log, "a", encoding="utf-8") as master_log:
                    master_log.write(f"Successfully saved chunked output file\n")
                    # Check for any papers with <NONE> implications
                    none_impl_papers = [code for code, paper in processed_papers_for_this_file.items() 
                                        if paper["[Implications]"] == "<NONE>"]
                    if none_impl_papers:
                        master_log.write(f"WARNING: {len(none_impl_papers)} papers have <NONE> for [Implications] section:\n")
                        for code in none_impl_papers[:10]:  # Show first 10 only if many
                            master_log.write(f"  - {code}\n")
                        if len(none_impl_papers) > 10:
                            master_log.write(f"  - ... and {len(none_impl_papers) - 10} more\n")
            except Exception as write_err:
                logger.error(f"Failed to write chunked file {output_file_path}: {write_err}")
                
                with open(master_debug_log, "a", encoding="utf-8") as master_log:
                    master_log.write(f"ERROR: Failed to write output file: {str(write_err)}\n")
            
            metrics["input_files_processed"] += 1
        except FileNotFoundError:
            logger.error(f"Input JSON file not found: {input_file_path}")
            
            with open(master_debug_log, "a", encoding="utf-8") as master_log:
                master_log.write(f"ERROR: Input file not found\n")
        except Exception as e:
            logger.error(f"Unexpected error processing input file {json_file}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            with open(master_debug_log, "a", encoding="utf-8") as master_log:
                master_log.write(f"ERROR: Unexpected error: {str(e)}\n")
                master_log.write(f"Traceback: {traceback.format_exc()}\n")
    
    # Save all processed papers from all files into one aggregated file
    all_papers_file_path = os.path.join(chunked_dir, "all_papers_chunked.txt")
    
    with open(master_debug_log, "a", encoding="utf-8") as master_log:
        master_log.write(f"\n=== Saving aggregated output to: {all_papers_file_path} ===\n")
    
    try:
        with open(all_papers_file_path, "w", encoding="utf-8") as f_all:
            json.dump(all_processed_papers, f_all, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved aggregated chunked sections for all papers to {all_papers_file_path}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"Successfully saved aggregated output file with {len(all_processed_papers)} papers\n")
            
            # Final check for papers with <NONE> implications
            none_impl_papers = [code for code, paper in all_processed_papers.items() 
                               if paper["[Implications]"] == "<NONE>"]
            if none_impl_papers:
                master_log.write(f"\nFINAL CHECK: {len(none_impl_papers)} papers have <NONE> for [Implications] section\n")
                master_log.write("This could indicate an issue with the extraction process\n")
    except Exception as write_err:
        logger.error(f"Failed to write aggregated chunked file {all_papers_file_path}: {write_err}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"ERROR: Failed to write aggregated output file: {str(write_err)}\n")
    
    # Finalize and save metrics
    metrics["end_time"] = datetime.now().isoformat()
    try:
        start = datetime.fromisoformat(metrics["start_time"])
        end = datetime.fromisoformat(metrics["end_time"])
        metrics["total_time_seconds"] = (end - start).total_seconds()
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"\n=== Processing completed ===\n")
            master_log.write(f"End time: {metrics['end_time']}\n")
            master_log.write(f"Total processing time: {metrics['total_time_seconds']:.2f} seconds\n")
    except Exception as time_err:
        logger.error(f"Error calculating total processing time: {time_err}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"ERROR calculating processing time: {str(time_err)}\n")
    
    # Save detailed metrics
    metrics_file_path = os.path.join(metrics_dir, "chunking_metrics.json")
    
    with open(master_debug_log, "a", encoding="utf-8") as master_log:
        master_log.write(f"\n=== Saving metrics to: {metrics_file_path} ===\n")
    
    try:
        with open(metrics_file_path, "w", encoding="utf-8") as f_metrics:
            json.dump(metrics, f_metrics, indent=2)
        
        logger.info(f"Chunking metrics saved to {metrics_file_path}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"Successfully saved metrics file\n")
    except Exception as write_err:
        logger.error(f"Failed to write metrics file {metrics_file_path}: {write_err}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"ERROR: Failed to write metrics file: {str(write_err)}\n")
    
    # Save a human-readable metrics summary
    metrics_summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
    
    with open(master_debug_log, "a", encoding="utf-8") as master_log:
        master_log.write(f"\n=== Saving human-readable summary to: {metrics_summary_path} ===\n")
    
    try:
        with open(metrics_summary_path, "w", encoding="utf-8") as f_summary:
            f_summary.write("=== Abstract Chunking Processing Summary ===\n\n")
            f_summary.write(f"Start time: {metrics['start_time']}\n")
            f_summary.write(f"End time: {metrics['end_time']}\n")
            f_summary.write(f"Total processing time: {metrics.get('total_time_seconds', 'N/A'):.2f} seconds\n\n")
            
            f_summary.write("=== Overall Statistics ===\n")
            f_summary.write(f"Input files processed: {metrics['input_files_processed']}\n")
            f_summary.write(f"Papers processed: {metrics['papers_processed']}\n")
            f_summary.write(f"Sections extracted: {metrics['sections_extracted']}\n\n")
            
            f_summary.write("=== Success/Failure Statistics ===\n")
            f_summary.write(f"Successful extractions: {metrics['successful_extractions']}\n")
            f_summary.write(f"Failed extractions: {metrics['failed_extractions']}\n")
            f_summary.write(f"Missing raw files: {metrics['missing_raw_files']}\n\n")
            
            # Error details by type
            f_summary.write("=== Error Details by Type ===\n")
            for key in sorted([k for k in metrics.keys() if k.startswith("error_type_")]):
                error_type = key.replace("error_type_", "")
                count = metrics[key]
                f_summary.write(f"{error_type}: {count}\n")
            f_summary.write("\n")
            
            # Failed files summary
            if metrics["failed_files"]:
                f_summary.write(f"=== Raw Files with Failed Extraction ({len(metrics['failed_files'])}) ===\n")
                for failed_file in sorted(metrics["failed_files"])[:20]:  # Show only first 20 if there are many
                    f_summary.write(f"  - {failed_file}\n")
                if len(metrics["failed_files"]) > 20:
                    f_summary.write(f"  - ... and {len(metrics['failed_files']) - 20} more\n")
                f_summary.write("\n")
            else:
                f_summary.write("=== No raw files failed extraction ===\n\n")
            
            # Add special section for implications stats
            f_summary.write("=== Implications Section Analysis ===\n")
            none_impl_papers = [code for code, paper in all_processed_papers.items() 
                               if paper["[Implications]"] == "<NONE>"]
            total_papers = len(all_processed_papers)
            impl_percentage = 100 - (len(none_impl_papers) / total_papers * 100) if total_papers > 0 else 0
            
            f_summary.write(f"Total papers: {total_papers}\n")
            f_summary.write(f"Papers with extracted Implications: {total_papers - len(none_impl_papers)} ({impl_percentage:.1f}%)\n")
            f_summary.write(f"Papers missing Implications: {len(none_impl_papers)} ({100 - impl_percentage:.1f}%)\n")
        
        logger.info(f"Human-readable metrics summary saved to {metrics_summary_path}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"Successfully saved human-readable summary\n")
    except Exception as write_err:
        logger.error(f"Failed to write metrics summary file {metrics_summary_path}: {write_err}")
        
        with open(master_debug_log, "a", encoding="utf-8") as master_log:
            master_log.write(f"ERROR: Failed to write metrics summary: {str(write_err)}\n")
    
    # Log summary results
    logger.info("--- Processing Summary ---")
    logger.info(f"Input files processed: {metrics['input_files_processed']}")
    logger.info(f"Papers processed: {metrics['papers_processed']}")
    logger.info(f"Sections extracted: {metrics['sections_extracted']}")
    logger.info(f"Successful extractions: {metrics['successful_extractions']}")
    logger.info(f"Failed extractions: {metrics['failed_extractions']}")
    logger.info(f"Missing raw files: {metrics['missing_raw_files']}")
    logger.info(f"Total processing time: {metrics['total_time_seconds']:.2f} seconds" if isinstance(metrics.get('total_time_seconds'), float) else "N/A")
    
    with open(master_debug_log, "a", encoding="utf-8") as master_log:
        master_log.write("\n=== PROCESSING COMPLETE ===\n")
    
    return metrics


def main():
    """Main function to parse arguments and run the abstract chunking process."""
    parser = argparse.ArgumentParser(
        description="Extract chunked sections from raw LLM output files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--domains", type=str, required=True,
                        help="Domains to process (e.g., 'ACL DH CSCW BIOINFO JMIR'). Used to find input/output subdirectories.")
    parser.add_argument("--base-dir", type=str, default="./output/event_segmentation",
                        help="Base directory where prompt/model/domain output structures reside.")
    parser.add_argument("--input-dir", type=str, default="./data/input",
                        help="Base directory containing domain-specific input JSON files (e.g., ./SciEvent_data/raw/ACL).")
    parser.add_argument("--prompt-template", type=str, default="Zeroshot",
                        help="Name of the prompt template used (subdirectory name under base-dir).")
    parser.add_argument("--model-name", type=str, default="gpt-4.1",# DeepSeek-R1-Distill-Llama-8B Qwen2.5-7B-Instruct Meta-Llama-3-8B-Instruct
                        help="Name of the model used (subdirectory name under prompt-template).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for more verbose output.")
    
    args = parser.parse_args()
    
    # Parse the domains list using space as separator instead of comma
    domains = args.domains.split()
    
    # Process each domain
    all_metrics = {}
    overall_start_time = datetime.now()
    
    for domain in domains:
        print(f"\n=================================================")
        print(f"Starting processing for domain: {domain}")
        print(f"=================================================\n")
        
        # Construct specific paths based on arguments
        domain_input_dir = os.path.join(args.input_dir, domain)
        output_base_dir = os.path.join(args.base_dir, args.model_name, args.prompt_template)
        raw_output_dir = os.path.join(output_base_dir, "raw_output")
        log_dir = os.path.join(output_base_dir, "logs")
        
        # Set up logging
        logger = setup_logging(log_dir)
        
        # Set debug level if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        logger.info("=================================================")
        logger.info(f"Starting Abstract Chunking Process")
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
            logger.error("Please ensure the input data exists. Skipping domain.")
            print(f"Error: Input directory not found for domain '{domain}'. Skipping.")
            continue
        
        if not os.path.exists(raw_output_dir):
            logger.error(f"Raw output directory not found: {raw_output_dir}")
            logger.error("Please ensure the raw LLM output files are present. Skipping domain.")
            print(f"Error: Raw output directory not found for domain '{domain}'. Skipping.")
            continue
        
        # Start Processing
        try:
            metrics = process_files(domain_input_dir, raw_output_dir, logger)
            all_metrics[domain] = metrics
            
            logger.info("=================================================")
            logger.info(f"Abstract Chunking Process Finished for domain: {domain}")
            # Log key metrics for this domain
            logger.info(f"Domain Metrics Summary:")
            logger.info(f"  Input Files Processed: {metrics.get('input_files_processed', 'N/A')}")
            logger.info(f"  Papers Processed: {metrics.get('papers_processed', 'N/A')}")
            logger.info(f"  Sections Extracted: {metrics.get('sections_extracted', 'N/A')}")
            logger.info(f"  Successful Extractions: {metrics.get('successful_extractions', 'N/A')}")
            logger.info(f"  Failed Extractions: {metrics.get('failed_extractions', 'N/A')}")
            logger.info(f"  Domain Processing Time: {metrics.get('total_time_seconds', 'N/A'):.2f} seconds" if isinstance(metrics.get('total_time_seconds'), float) else "N/A")
            logger.info("=================================================")
            
            print(f"Completed processing domain: {domain}")
            print(f"  Papers Processed: {metrics.get('papers_processed', 'N/A')}")
            print(f"  Successful Extractions: {metrics.get('successful_extractions', 'N/A')}")
            print(f"  Failed Extractions: {metrics.get('failed_extractions', 'N/A')}")
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"Error processing domain {domain}: {str(e)}")
    
    # Calculate and display overall metrics
    overall_end_time = datetime.now()
    overall_time_seconds = (overall_end_time - overall_start_time).total_seconds()
    
    # Create summary directory in base_dir
    summary_dir = os.path.join(args.base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save all domains metrics to a file
    all_metrics_file = os.path.join(summary_dir, f"all_domains_metrics_{args.model_name}.json")
    try:
        with open(all_metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
    except Exception as e:
        print(f"Error saving all domains metrics: {str(e)}")
    
    # Create a human-readable summary
    summary_file = os.path.join(summary_dir, f"all_domains_summary_{args.model_name}.txt")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=================================================\n")
            f.write("Abstract Chunking - Multi-Domain Processing Summary\n")
            f.write("=================================================\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Prompt Template: {args.prompt_template}\n")
            f.write(f"Domains Processed: {', '.join(domains)}\n")
            f.write(f"Start Time: {overall_start_time.isoformat()}\n")
            f.write(f"End Time: {overall_end_time.isoformat()}\n")
            f.write(f"Total Processing Time: {overall_time_seconds:.2f} seconds\n\n")
            
            # Aggregate metrics
            total_papers = sum(metrics.get('papers_processed', 0) for metrics in all_metrics.values())
            total_sections = sum(metrics.get('sections_extracted', 0) for metrics in all_metrics.values())
            total_success = sum(metrics.get('successful_extractions', 0) for metrics in all_metrics.values())
            total_failures = sum(metrics.get('failed_extractions', 0) for metrics in all_metrics.values())
            
            f.write("=== Overall Statistics ===\n")
            f.write(f"Total Papers Processed: {total_papers}\n")
            f.write(f"Total Sections Extracted: {total_sections}\n")
            f.write(f"Total Successful Extractions: {total_success}\n")
            f.write(f"Total Failed Extractions: {total_failures}\n\n")
            
            f.write("=== Domain-Specific Statistics ===\n")
            for domain, metrics in all_metrics.items():
                f.write(f"\n== Domain: {domain} ==\n")
                f.write(f"  Input Files Processed: {metrics.get('input_files_processed', 'N/A')}\n")
                f.write(f"  Papers Processed: {metrics.get('papers_processed', 'N/A')}\n")
                f.write(f"  Sections Extracted: {metrics.get('sections_extracted', 'N/A')}\n")
                f.write(f"  Successful Extractions: {metrics.get('successful_extractions', 'N/A')}\n")
                f.write(f"  Failed Extractions: {metrics.get('failed_extractions', 'N/A')}\n")
            f.write(f"  Domain Processing Time: {metrics.get('total_time_seconds', 'N/A'):.2f} seconds\n" if isinstance(metrics.get('total_time_seconds'), float) else "  Domain Processing Time: N/A\n")
            
            # Add domain-specific analysis of implications sections
            if "implications_analysis" in metrics:
                f.write(f"  Implications Analysis:\n")
                f.write(f"    Papers with Implications: {metrics['implications_analysis'].get('with_implications', 'N/A')}\n")
                f.write(f"    Papers without Implications: {metrics['implications_analysis'].get('without_implications', 'N/A')}\n")
    except Exception as e:
        print(f"Error creating summary file: {str(e)}")
    
    # Print overall summary to console
    print("\n=================================================")
    print("Abstract Chunking - Multi-Domain Processing Complete")
    print("=================================================")
    print(f"Domains Processed: {', '.join(domains)}")
    print(f"Total Papers Processed: {total_papers}")
    print(f"Total Sections Extracted: {total_sections}")
    print(f"Total Successful Extractions: {total_success}")
    print(f"Total Failed Extractions: {total_failures}")
    print(f"Total Processing Time: {overall_time_seconds:.2f} seconds")
    print(f"Summary saved to: {summary_file}")
    print("=================================================")


if __name__ == "__main__":
    main()