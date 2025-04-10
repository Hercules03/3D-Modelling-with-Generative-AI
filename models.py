"""
Pydantic models for data validation throughout the application
"""

from typing import List, Dict, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator
import datetime
import os

# Basic models

class KeywordData(BaseModel):
    """Model for extracted keywords"""
    core_type: str = Field(description="Main object type (e.g., 'table', 'gear')")
    modifiers: List[str] = Field(default_factory=list, description="Modifiers for the core type")
    compound_type: str = Field(default="", description="Combined description of the object")

    @field_validator('core_type')
    def core_type_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Core type cannot be empty")
        return v.strip()

class Parameter(BaseModel):
    """Model for OpenSCAD parameters"""
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value (as string)")
    comment: Optional[str] = Field(default=None, description="Optional parameter description")

class CodeMetrics(BaseModel):
    """Model for code analysis metrics"""
    line_count: int = Field(default=0, description="Number of lines in the code")
    module_count: int = Field(default=0, description="Number of modules in the code") 
    function_count: int = Field(default=0, description="Numbfield_validatorer of functions in the code")
    comment_lines: int = Field(default=0, description="Number of comment lines")
    operation_count: int = Field(default=0, description="Number of operations (union, difference, etc.)")
    complexity: Literal["SIMPLE", "MEDIUM", "COMPLEX"] = Field(
        default="SIMPLE", 
        description="Overall code complexity"
    )
    primary_style: Optional[str] = Field(
        default=None, 
        description="Primary coding style (Parametric, Modular, etc.)"
    )
    style_breakdown: Optional[Dict[str, int]] = Field(
        default=None,
        description="Breakdown of different coding styles and their prevalence"
    )

# Step-back analysis models

class StepBackAnalysis(BaseModel):
    """Model for step-back analysis results"""
    principles: List[str] = Field(
        default_factory=list,
        description="Core principles identified in the analysis"
    )
    abstractions: List[str] = Field(
        default_factory=list,
        description="Shape components or abstractions identified"
    )
    approach: List[str] = Field(
        default_factory=list,
        description="Implementation steps or approaches"
    )
    original_query: Optional[str] = Field(
        default=None,
        description="Original query that prompted this analysis"
    )
    keyword_analysis: Optional[KeywordData] = Field(
        default=None,
        description="Keywords extracted from the query"
    )

# Query analysis models

class QueryAnalysis(BaseModel):
    """Model for SCAD code retrieval query analysis"""
    search_strategy: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Strategy for searching the vector database"
    )
    enhanced_query: str = Field(
        description="Reformulated query optimized for finding similar examples"
    )
    important_attributes: List[str] = Field(
        default_factory=list,
        description="Important attributes for filtering results"
    )
    style_preference: str = Field(
        default="Parametric",
        description="Preferred SCAD coding style"
    )
    complexity: Literal["SIMPLE", "MEDIUM", "COMPLEX"] = Field(
        default="MEDIUM",
        description="Expected code complexity"
    )
    similarities_to_check: List[str] = Field(
        default_factory=list,
        description="Aspects of code implementation to match"
    )
    techniques_needed: List[str] = Field(
        default_factory=list,
        description="OpenSCAD techniques likely needed for implementation"
    )

# Example models

class ScoreBreakdown(BaseModel):
    """Model for example score breakdown"""
    final_score: float = Field(description="Overall relevance score")
    component_scores: Dict[str, float] = Field(
        description="Individual scores for different aspects"
    )
    step_back_details: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed scores for step-back analysis components"
    )
    matching_techniques: List[str] = Field(
        default_factory=list,
        description="Techniques that matched between query and example"
    )

class SimilarExample(BaseModel):
    """Model for a similar example from the knowledge base"""
    id: str = Field(description="Unique identifier for the example")
    description: str = Field(description="Description of the example")
    code: str = Field(description="OpenSCAD code for the example")
    score: float = Field(description="Similarity score to the query")
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Detailed breakdown of the similarity score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the example"
    )
    techniques_used: List[str] = Field(
        default_factory=list,
        description="OpenSCAD techniques used in this example"
    )
    code_metrics: Optional[CodeMetrics] = Field(
        default=None,
        description="Metrics for the code in this example"
    )
    primary_parameters: List[Parameter] = Field(
        default_factory=list,
        description="Primary parameters used in this example"
    )

# Export models

class ExportResult(BaseModel):
    """Model for the result of model export"""
    success: bool = Field(description="Whether export was successful")
    format: str = Field(description="Format the model was exported to")
    file_path: Optional[str] = Field(default=None, description="Path to the exported file")
    file_size: Optional[int] = Field(default=None, description="Size of the exported file in bytes")
    error: Optional[str] = Field(default=None, description="Error message if export failed")
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the model was exported"
    )
    
    def get_file_size(self) -> Optional[int]:
        """Get the size of the exported file"""
        if self.file_path and os.path.exists(self.file_path):
            return os.path.getsize(self.file_path)
        return None

# Generation result models

class ParameterAdjustment(BaseModel):
    """Model for a parameter adjustment"""
    parameter_name: str = Field(description="Name of the parameter")
    old_value: Any = Field(description="Original value of the parameter")
    new_value: Any = Field(description="New value of the parameter")
    explanation: Optional[str] = Field(default=None, description="Explanation for the change")

class ParameterTuningResult(BaseModel):
    """Model for parameter tuning results"""
    success: bool = Field(description="Whether parameter tuning was successful")
    parameters_changed: int = Field(default=0, description="Number of parameters changed")
    adjustments: List[ParameterAdjustment] = Field(default_factory=list, description="List of parameter adjustments")
    error: Optional[str] = Field(default=None, description="Error message if tuning failed")
    user_provided: bool = Field(default=False, description="Whether adjustments were user-provided")

class GenerationResult(BaseModel):
    """Model for the result of code generation"""
    success: bool = Field(description="Whether code generation was successful")
    code: Optional[str] = Field(default=None, description="Generated OpenSCAD code")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
    prompt: Optional[str] = Field(default=None, description="Prompt used for generation")
    add_to_kb: bool = Field(default=False, description="Whether to add to knowledge base")
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the code was generated"
    )
    exported_files: Optional[Dict[str, ExportResult]] = Field(
        default=None, 
        description="Results of model exports to different formats"
    )
    parameters_tuned: bool = Field(default=False, description="Whether parameters were tuned")
    parameter_tuning_result: Optional[ParameterTuningResult] = Field(
        default=None,
        description="Results of parameter tuning"
    )

# State models for LangGraph

class StateData(BaseModel):
    """Model for data in the LangGraph state"""
    input_text: Optional[str] = Field(default=None, description="User input text")
    extracted_keywords: Optional[KeywordData] = Field(
        default=None, 
        description="Extracted keywords from input"
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="Generated search queries"
    )
    search_results: Optional[Dict[str, List[Dict[str, str]]]] = Field(
        default=None,
        description="Raw search results"
    )
    filtered_search_results: Optional[Dict[str, List[Dict[str, str]]]] = Field(
        default=None,
        description="Filtered search results"
    )
    step_back_analysis: Optional[StepBackAnalysis] = Field(
        default=None,
        description="Step-back analysis results"
    )
    model_information: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Information about the model being generated"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debugging information"
    )
    hallucination_check: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hallucination check results"
    )
    analysis_grade: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Step-back analysis grade"
    )
    query_analysis: Optional[QueryAnalysis] = Field(
        default=None,
        description="Query analysis for retrieval"
    )
    similar_examples: Optional[List[SimilarExample]] = Field(
        default=None,
        description="Similar examples retrieved"
    )
    retrieved_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata retrieved with examples"
    )
    generated_code: Optional[GenerationResult] = Field(
        default=None,
        description="Generated code results"
    )
    
