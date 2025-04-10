"""
Model exporter for 3D-Modelling-with-Generative-AI
Provides functionality to export OpenSCAD models to different formats.
"""

import os
import tempfile
import subprocess
import logging
import shutil
from typing import Optional, Dict, List, Literal, Tuple, Union

logger = logging.getLogger(__name__)

# Define supported export formats
ExportFormat = Literal["stl", "off", "amf", "3mf", "csg", "dxf", "svg", "png"]

class ModelExporter:
    """Export OpenSCAD models to different formats"""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the model exporter
        
        Args:
            output_dir: Directory to save exported models (default: "output")
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logger.error(f"Error creating output directory: {str(e)}")
                raise
        
        # Check if the directory is writable
        if not os.access(output_dir, os.W_OK):
            error_msg = f"Output directory is not writable: {output_dir}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        
        # Check if OpenSCAD is installed
        if not self._check_openscad():
            error_msg = "OpenSCAD is not installed or not in PATH"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"ModelExporter initialized with output directory: {output_dir}")
    
    def _check_openscad(self) -> bool:
        """Check if OpenSCAD is installed and accessible"""
        try:
            result = subprocess.run(
                ["openscad", "--version"], 
                capture_output=True, 
                text=True,
                timeout=5  # Add timeout for safety
            )
            if result.returncode == 0:
                logger.info(f"OpenSCAD found: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"OpenSCAD returned error: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("OpenSCAD executable not found in PATH")
            return False
        except Exception as e:
            logger.warning(f"Error checking OpenSCAD: {str(e)}")
            return False
    
    def export_model(self, 
                    scad_code: str, 
                    filename: str = "model", 
                    export_format: ExportFormat = "stl", 
                    resolution: int = 100,
                    additional_params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Export SCAD code to different formats using OpenSCAD CLI
        
        Args:
            scad_code: OpenSCAD code to export
            filename: Name of the output file (without extension)
            export_format: Format to export to (stl, off, amf, 3mf, csg, dxf, svg, png)
            resolution: Resolution for curves ($fn value)
            additional_params: Additional parameters to pass to OpenSCAD
            
        Returns:
            Path to the exported file, or None if export failed
        """
        tmp_path = None
        try:
            # Make sure code is not empty
            if not scad_code or not scad_code.strip():
                logger.error("Cannot export empty SCAD code")
                return None
                
            # Ensure valid filename (remove problematic characters)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            if safe_filename != filename:
                logger.warning(f"Sanitized filename from '{filename}' to '{safe_filename}'")
                filename = safe_filename
            
            # Save SCAD code to temporary file
            with tempfile.NamedTemporaryFile(suffix=".scad", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(scad_code.encode('utf-8'))
                logger.debug(f"Saved SCAD code to temporary file: {tmp_path}")
            
            # Verify the temp file was created and contains content
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                logger.error("Failed to create temporary file or file is empty")
                return None
            
            # Create output path
            output_path = os.path.join(self.output_dir, f"{filename}.{export_format}")
            
            # Make sure output directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                logger.info(f"Created output directory: {self.output_dir}")
            
            # Prepare OpenSCAD command
            cmd = ["openscad", f"-o{output_path}", f"-D$fn={resolution}", tmp_path]
            
            # Add additional parameters if provided
            if additional_params:
                for key, value in additional_params.items():
                    cmd.append(f"-D{key}={value}")
            
            # Log the command
            logger.info(f"Running OpenSCAD command: {' '.join(cmd)}")
            
            # Run OpenSCAD
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120  # Add timeout (2 minutes) for safety
            )
            
            # Check if successful
            if process.returncode == 0:
                # Verify the file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully exported model to: {output_path}")
                    return output_path
                else:
                    logger.error(f"Export file not created or is empty: {output_path}")
                    return None
            else:
                logger.error(f"OpenSCAD export failed with error: {process.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("OpenSCAD process timed out")
            return None
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None
        finally:
            # Clean up temporary file
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.debug(f"Removed temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    def export_multiple_formats(self, 
                             scad_code: str, 
                             filename: str = "model",
                             formats: List[ExportFormat] = ["stl"],
                             resolution: int = 100,
                             additional_params: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
        """Export SCAD code to multiple formats
        
        Args:
            scad_code: OpenSCAD code to export
            filename: Base name for output files (without extension)
            formats: List of formats to export to
            resolution: Resolution for curves ($fn value)
            additional_params: Additional parameters to pass to OpenSCAD
            
        Returns:
            Dictionary mapping format to output path (or None if failed)
        """
        results = {}
        
        for fmt in formats:
            output_path = self.export_model(
                scad_code=scad_code,
                filename=filename,
                export_format=fmt,
                resolution=resolution,
                additional_params=additional_params
            )
            results[fmt] = output_path
        
        return results
    
    def generate_preview(self, 
                    scad_code: str, 
                    filename: str = "preview", 
                    resolution: int = 100, 
                    camera_params: str = "0,0,0,55,0,25,140",
                    width: int = 800, 
                    height: int = 600) -> Optional[str]:
        """Generate a PNG preview of the model.
        
        Args:
            scad_code (str): The OpenSCAD code to render
            filename (str): Base name for the output file (without extension)
            resolution (int): Resolution for curves ($fn value)
            camera_params (str): Camera position for the render in format "translatex,y,z,rotx,y,z,distance"
            width (int): Image width in pixels
            height (int): Image height in pixels
            
        Returns:
            Optional[str]: Path to the generated preview image or None if failed
        """
        tmp_path = None
        try:
            # Make sure code is not empty
            if not scad_code or not scad_code.strip():
                logger.error("Cannot generate preview from empty SCAD code")
                return None
                
            # Ensure valid filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            if safe_filename != filename:
                logger.warning(f"Sanitized filename from '{filename}' to '{safe_filename}'")
                filename = safe_filename
            
            # Save SCAD code to temporary file
            with tempfile.NamedTemporaryFile(suffix=".scad", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(scad_code.encode('utf-8'))
                logger.debug(f"Saved SCAD code to temporary file: {tmp_path}")
            
            # Create output path
            output_path = os.path.join(self.output_dir, f"{filename}.png")
            
            # Make sure output directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                logger.info(f"Created output directory: {self.output_dir}")
            
            # Prepare OpenSCAD command for preview
            cmd = [
                "openscad", 
                f"-o{output_path}", 
                f"--camera={camera_params}",
                f"--imgsize={width},{height}",
                f"-D$fn={resolution}", 
                tmp_path
            ]
            
            # Log the command
            logger.info(f"Running OpenSCAD preview command: {' '.join(cmd)}")
            
            # Run OpenSCAD
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120  # Add timeout (2 minutes) for safety
            )
            
            # Check if successful
            if process.returncode == 0:
                # Verify the file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully generated preview: {output_path}")
                    return output_path
                else:
                    logger.error(f"Preview file not created or is empty: {output_path}")
                    return None
            else:
                logger.error(f"OpenSCAD preview failed with error: {process.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("OpenSCAD process timed out")
            return None
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            return None
        finally:
            # Clean up temporary file
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.debug(f"Removed temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

    def export_from_file(self,
                      scad_file: str,
                      output_filename: Optional[str] = None,
                      export_format: ExportFormat = "stl",
                      resolution: int = 100,
                      additional_params: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Export an existing SCAD file to a different format
        
        Args:
            scad_file: Path to the SCAD file
            output_filename: Name for the output file (without extension, defaults to input filename)
            export_format: Format to export to
            resolution: Resolution for curves ($fn value)
            additional_params: Additional parameters to pass to OpenSCAD
            
        Returns:
            Path to the exported file, or None if export failed
        """
        try:
            # Check if input file exists
            if not os.path.exists(scad_file):
                logger.error(f"Input SCAD file not found: {scad_file}")
                return None
            
            # Get filename if not provided
            if output_filename is None:
                output_filename = os.path.splitext(os.path.basename(scad_file))[0]
            
            # Create output path
            output_path = os.path.join(self.output_dir, f"{output_filename}.{export_format}")
            
            # Prepare OpenSCAD command
            cmd = ["openscad", f"-o{output_path}", f"-D$fn={resolution}", scad_file]
            
            # Add additional parameters if provided
            if additional_params:
                for key, value in additional_params.items():
                    cmd.append(f"-D{key}={value}")
            
            # Log the command
            logger.info(f"Running OpenSCAD command: {' '.join(cmd)}")
            
            # Run OpenSCAD
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if successful
            if process.returncode == 0:
                logger.info(f"Successfully exported model to: {output_path}")
                return output_path
            else:
                logger.error(f"OpenSCAD export failed with error: {process.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None