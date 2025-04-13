#!/usr/bin/env python3
"""
Standalone script to export OpenSCAD models to various formats
"""

import os
import tempfile
import subprocess
import argparse
from typing import Optional, List

def export_model(
    scad_file: str, 
    output_dir: str = "output", 
    filename: str = "model",
    formats: List[str] = ["stl"],
    resolution: int = 100
) -> dict:
    """Export a SCAD file to specified formats
    
    Args:
        scad_file: Path to OpenSCAD file
        output_dir: Directory to save exported models
        filename: Base name for output files (without extension)
        formats: List of formats to export to
        resolution: Resolution for curves ($fn value)
        
    Returns:
        Dictionary with export results
    """
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Dictionary to store results
    results = {}
    
    # Check if input file exists
    if not os.path.exists(scad_file):
        print(f"Error: SCAD file not found: {scad_file}")
        return {"success": False, "error": "SCAD file not found"}
    
    # Export to each format
    for fmt in formats:
        print(f"Exporting to {fmt} format...")
        
        # Create output path
        output_path = os.path.join(output_dir, f"{filename}.{fmt}")
        
        # Prepare OpenSCAD command
        cmd = ["openscad", f"-o{output_path}", f"-D$fn={resolution}", scad_file]
        
        try:
            # Run OpenSCAD
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if successful
            if process.returncode == 0:
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
                results[fmt] = {
                    "success": True,
                    "file_path": output_path,
                    "file_size": file_size
                }
                print(f"  Success! Exported to {output_path}")
                if file_size:
                    print(f"  File size: {file_size} bytes")
            else:
                results[fmt] = {
                    "success": False,
                    "error": process.stderr
                }
                print(f"  Failed! Error: {process.stderr}")
                
        except Exception as e:
            results[fmt] = {
                "success": False,
                "error": str(e)
            }
            print(f"  Error: {str(e)}")
    
    return {
        "success": any(result["success"] for result in results.values()),
        "results": results
    }

def export_from_code(
    scad_code: str, 
    output_dir: str = "output", 
    filename: str = "model",
    formats: List[str] = ["stl"],
    resolution: int = 100
) -> dict:
    """Export SCAD code to various formats
    
    Args:
        scad_code: OpenSCAD code as a string
        output_dir: Directory to save exported models
        filename: Base name for output files (without extension)
        formats: List of formats to export to
        resolution: Resolution for curves ($fn value)
        
    Returns:
        Dictionary with export results
    """
    try:
        # Save SCAD code to temporary file
        with tempfile.NamedTemporaryFile(suffix=".scad", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(scad_code.encode())
            print(f"Saved SCAD code to temporary file: {tmp_path}")
        
        # Export using the temporary file
        results = export_model(
            scad_file=tmp_path,
            output_dir=output_dir,
            filename=filename,
            formats=formats,
            resolution=resolution
        )
        
        return results
    
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Removed temporary file: {tmp_path}")

def main():
    """Main function to parse arguments and export model"""
    parser = argparse.ArgumentParser(description="Export OpenSCAD models to various formats")
    
    # Input options - either file or code
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", help="Path to OpenSCAD file")
    input_group.add_argument("-c", "--code", help="OpenSCAD code (as a string)")
    input_group.add_argument("--code-file", help="File containing OpenSCAD code")
    
    # Export options
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("-n", "--name", default="model", help="Base filename for output")
    parser.add_argument("--formats", default="stl", help="Comma-separated list of export formats")
    parser.add_argument("-r", "--resolution", type=int, default=100, help="Resolution for curves ($fn value)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process formats
    formats = [fmt.strip() for fmt in args.formats.split(",")]
    
    # Export based on input type
    if args.file:
        print(f"Exporting from SCAD file: {args.file}")
        result = export_model(
            scad_file=args.file,
            output_dir=args.output_dir,
            filename=args.name,
            formats=formats,
            resolution=args.resolution
        )
    elif args.code:
        print("Exporting from SCAD code")
        result = export_from_code(
            scad_code=args.code,
            output_dir=args.output_dir,
            filename=args.name,
            formats=formats,
            resolution=args.resolution
        )
    elif args.code_file:
        print(f"Reading SCAD code from file: {args.code_file}")
        with open(args.code_file, "r") as f:
            scad_code = f.read()
        result = export_from_code(
            scad_code=scad_code,
            output_dir=args.output_dir,
            filename=args.name,
            formats=formats,
            resolution=args.resolution
        )
    
    # Print summary
    if result["success"]:
        print("\nExport completed successfully!")
    else:
        print("\nExport failed!")
    
    for fmt, fmt_result in result["results"].items():
        success = fmt_result["success"]
        status = "Success" if success else "Failed"
        print(f"  {fmt}: {status}")
        if success:
            print(f"    File: {fmt_result['file_path']}")
        else:
            print(f"    Error: {fmt_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()