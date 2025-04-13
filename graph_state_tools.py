from typing_extensions import TypedDict
from typing import Annotated, Optional, Dict, Any, List, Literal
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import *
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from myAPI import TAVILY_API_KEY
from step_back_analyzer import StepBackAnalyzer
from conversation_logger import ConversationLogger
from pydantic import BaseModel, Field, field_validator
import re
import datetime

# Import our new Pydantic models
from models import KeywordData, StepBackAnalysis, QueryAnalysis, GenerationResult, StateData
# Tavily API Configuration
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class State(TypedDict):
    messages: Annotated[list, add_messages]
    input_text: Optional[str]
    extracted_keywords: Optional[Dict[str, Any]]
    search_queries: Optional[List[str]]
    search_results: Optional[Dict[str, List[Dict[str, str]]]]
    filtered_search_results: Optional[Dict[str, List[Dict[str, str]]]]
    step_back_analysis: Optional[Dict[str, Any]]
    model_information: Optional[Dict[str, Any]]
    debug_info: Optional[Dict[str, Any]]
    hallucination_check: Optional[Dict[str, Any]]
    analysis_grade: Optional[Dict[str, Any]] 
    query_analysis: Optional[Dict[str, Any]]
    similar_examples: Optional[Dict[str, Any]]
    retrieved_metadata: Optional[Dict[str, Any]]
    generated_code: Optional[Dict[str, Any]]

class GradeWebContent(BaseModel):
    """Binary score for relevance check on retrieved web content."""
    binary_score: str = Field(
        description="Web content is relevant to the question, 'yes' or 'no'"
    )
    
    @field_validator('binary_score')
    def validate_binary_score(cls, v):
        v = v.lower().strip()
        if v not in ['yes', 'no']:
            raise ValueError("binary_score must be 'yes' or 'no'")
        return v
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in step back analysis."""
    binary_score: Literal["yes", "no"] = Field(
        description="Analysis is grounded in the facts, 'yes' or 'no'"
    )
    
    @field_validator('binary_score')
    def validate_binary_score(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v not in ['yes', 'no']:
                raise ValueError("binary_score must be 'yes' or 'no'")
        return v

    
class GradeStepBackAnalysis(BaseModel):
    """Rating scale to assess if step back analysis properly addresses the 3D modeling task."""
    rating: int = Field(
        description="Quality rating of the analysis on a scale of 0-10 (0=poor, 10=excellent)",
        ge=0,  # greater than or equal to 0
        le=10  # less than or equal to 10
    )
    feedback: str = Field(
        description="Feedback on how to improve the analysis",
        min_length=10  # Ensure feedback is not too short
    )
    
    @field_validator('rating')
    def validate_rating(cls, v):
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError("Rating must be an integer between 0 and 10")
        return v
class SCADQueryAnalysis(BaseModel):
    """Analysis of a query for finding similar SCAD code examples."""
    search_strategy: Literal["semantic", "keyword", "hybrid"] = Field(
        description="Best strategy to use: 'semantic' (similar meaning), 'keyword' (exact matches), or 'hybrid' (both)"
    )
    enhanced_query: str = Field(
        description="Reformulated query optimized for finding similar SCAD code examples",
        min_length=3
    )
    important_attributes: List[str] = Field(
        description="List of attributes relevant for filtering SCAD code (dimensions, mechanics, etc.)",
        default_factory=list
    )
    style_preference: str = Field(
        description="SCAD coding style preference (e.g., 'Modular', 'Parametric', 'Functional', etc.)"
    )
    complexity: Literal["SIMPLE", "MEDIUM", "COMPLEX"] = Field(
        description="Complexity of the required SCAD code (SIMPLE, MEDIUM, COMPLEX)"
    )
    code_similarities: List[str] = Field(
        description="Aspects of SCAD code implementation to match (module structure, algorithms, etc.)",
        default_factory=list
    )
    
    @field_validator('search_strategy')
    def validate_search_strategy(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v not in ['semantic', 'keyword', 'hybrid']:
                raise ValueError("search_strategy must be 'semantic', 'keyword', or 'hybrid'")
        return v
        
    @field_validator('complexity')
    def validate_complexity(cls, v):
        if isinstance(v, str):
            v = v.upper().strip()
            if v not in ['SIMPLE', 'MEDIUM', 'COMPLEX']:
                raise ValueError("complexity must be 'SIMPLE', 'MEDIUM', or 'COMPLEX'")
        return v


# Define the node to process user input
def process_input(state: State) -> Dict:
    """Process the user input and prepare it for keyword extraction"""
    messages = state.get("messages", [])
    
    # Get the last user message
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_input = message.content
            break
    else:
        # Default case if no human message is found
        user_input = ""
    
    return {
        "input_text": user_input,
        "debug_info": {"stage": "input_processing", "user_input": user_input}
    }
    
# Define the node to extract keywords
def extract_keywords(state: State) -> Dict:
    """Extract keywords from the input text"""
    input_text = state.get("input_text", "")
    debug_info = state.get("debug_info", {})
    feedback = False
    
    print("="*25,"Performing Keyword Extraction","="*25)
    print()
    
    keyword_extractor_llm = ChatOllama(
        model="keyword-extractor:latest",
        temperature=0.0,
        base_url="http://localhost:11434"
    )
    
    while not feedback: 
        # If there's feedback, incorporate it into the prompt
        if feedback:
            formatted_prompt = KEYWORD_EXTRACTOR_PROMPT.replace("<<query>>", input_text) + f"\n\nPlease consider this feedback for improvement: {feedback}"
        else:
            formatted_prompt = KEYWORD_EXTRACTOR_PROMPT.replace("<<query>>", input_text)
    
        print(f"\nFormatted prompt:\n{formatted_prompt}")
        print('-'*50)
            
        messages = [
            SystemMessage(content=KEYWORD_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=formatted_prompt)
        ]
            
        # Get response from the LLM
        response = keyword_extractor_llm.invoke(messages)
            
        content = response.content.strip()
        print(f"\nRaw LLM Response content:\n{content}\n")
        print('-'*50)
            
        # Parse the entire response as JSON first
        start_marker = "```json"
        end_marker = "```"
        json_content = ""
        
        # Find the JSON content between the markers
        start_index = content.find(start_marker)
        if start_index != -1:
            end_index = content.find(end_marker, start_index + len(start_marker))
            if end_index != -1:
                json_content = content[start_index + len(start_marker):end_index]
        extracted_keywords = json.loads(json_content)
        
        # Request human approval and get feedback
        print('-'*50)
        print(f"Extracted Keywords:")
        print(f"Core Type: {extracted_keywords.get('core_type', 'None')}")
        print(f"Modifiers: {', '.join(extracted_keywords.get('modifiers', []))}")
        print(f"Compound Type: {extracted_keywords.get('compound_type', 'None')}")
        print('-'*50)
        approval = input("Do you approve these keywords? (yes/no)")
        if approval.lower() == "yes":
            feedback = True
        else:
            feedback = False
            
    # Create updated state
    updated_state = {
        "extracted_keywords": extracted_keywords,
        "debug_info": {
            **debug_info,
            "stage": "keyword_extraction",
            "extracted_keywords": extracted_keywords,
        }
    }
    
    return updated_state
        
        
# Define the node to create search queries
def create_search_queries(state: State) -> Dict:
    """Create search queries based on the extracted keywords"""
    extracted_keywords = state.get("extracted_keywords", {})
    debug_info = state.get("debug_info", {})
    
    print("\n" + "="*25 + " Creating Search Queries " + "="*25)
    print(f"\nUsing keywords: {extracted_keywords}")
    
    core_type = extracted_keywords.get("core_type", "")
    modifiers = extracted_keywords.get("modifiers", [])
    compound_type = extracted_keywords.get("compound_type", "")
    
    # Create multiple search queries for different aspects
    search_queries = []
    
    # Use compound type if available, otherwise combine modifiers and core type
    main_object = compound_type if compound_type else f"{' '.join(modifiers)} {core_type}".strip()
    
    print(f"\nGenerating search queries for: {main_object}")
    
    if main_object:
        # Search for shape and geometry
        search_queries.append(f"{main_object} basic shape geometry form proportions 3D model")
        
        # Search for components and structure
        search_queries.append(f"{main_object} components assembly structure breakdown 3D")
        
        # Search for purpose and function
        search_queries.append(f"{main_object} purpose function usage design considerations")
        
        # Search for physical constraints and manufacturing requirements
        search_queries.append(f"{main_object} 3D Printing constraints physical limitations materials")
        
        # Search for technical 3D modeling requirements
        search_queries.append(f"{main_object} 3D model technical requirements polygon count formats specifications")
    
    print("\nGenerated Search Queries:")
    for query in search_queries:
        print(f"- {query}")
    print()
    
    # Create updated state
    updated_state = {
        **state,  # Preserve existing state
        "search_queries": search_queries,
        "debug_info": {
            **debug_info,
            "stage": "query_creation",
            "search_queries": search_queries,
            "main_object": main_object
        }
    }
    
    print("\nMoving to web search phase...")
    return updated_state
    
# Define the node to perform search
def perform_search(state: State) -> Dict:
    """Perform searches using the generated queries"""
    search_tool = TavilySearchResults(max_results=1)
    search_queries = state.get("search_queries", [])
    debug_info = state.get("debug_info", {})
    
    print("="*25,"Performing Search","="*25)
    print()
    print(f"Search Queries: {search_queries}")
    print()
    print("="*50)
    
    search_results = {}
    
    # Perform searches for each query
    for i, query in enumerate(search_queries):
        print(f"\nProcessing query {i+1}/{len(search_queries)}: {query}")
        try:
            category = f"query_{i+1}"
            results = search_tool.invoke(query)
            
            # Ensure results is properly formatted
            if not isinstance(results, list):
                print(f"Warning: Unexpected result format: {type(results)}")
                results = []
                
            # Process each result to ensure it's a dictionary
            formatted_results = []
            for r in results:
                if isinstance(r, dict):
                    formatted_results.append(r)
                elif isinstance(r, str):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(r)
                        if isinstance(parsed, dict):
                            formatted_results.append(parsed)
                        else:
                            # Create a dictionary with the string as content
                            formatted_results.append({"title": "Search Result", "url": "", "content": r})
                    except json.JSONDecodeError:
                        # Just use the string as content
                        formatted_results.append({"title": "Search Result", "url": "", "content": r})
                else:
                    print(f"Warning: Skipping result of unexpected type: {type(r)}")
            
            search_results[category] = formatted_results
            print(f"Found {len(formatted_results)} formatted results for query {i+1}")
        except Exception as e:
            print(f"Error searching for query {i+1}: {str(e)}")
            continue
    
    # Print formatted search results
    print("\nSearch Results Summary:")
    for category, results in search_results.items():
        print(f"\n{category}:")
        for result in results:
            print("\n" + "-"*50)
            print(f"Title: {result.get('title', 'No title')}")
            print(f"URL: {result.get('url', 'No URL')}")
            print(f"Content length: {len(result.get('content', ''))}")
    
    updated_debug_info = {
        **debug_info,
        "stage": "search",
        "search_completed": True,
        "num_queries": len(search_queries),
        "num_results": sum(len(results) for results in search_results.values())
    }
    
    return {
        "search_results": search_results,
        "debug_info": updated_debug_info
    }
    
# Define the node to grade web content
def grade_web_content(state: State) -> Dict:
    """Grade the web content based on relevance to the question"""
    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", {})
    debug_info = state.get("debug_info", {})
    extracted_keywords = state.get("extracted_keywords", {})
    
    print("="*25,"Grading Web Content","="*25)
    print()
    
    web_content_grader_llm = ChatOllama(
        model="llama3.2:3b-instruct-q4_K_M",
        temperature=0.0,
        base_url="http://localhost:11434"
    )
    
    structured_llm_grader = web_content_grader_llm.with_structured_output(GradeWebContent)
    
    # Create a simpler prompt that just asks for yes/no
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_CONTENT_GRADER_PROMPT),
        ("human", "Retrieved web content: \n\n {content} \n\n Search query: {search_query} \n\n Object: {object_type} \n\nIs this content relevant? Answer with just 'yes' or 'no'.")
    ])
    
    # Create the grader chain
    web_content_grader = grade_prompt | structured_llm_grader
    
    # Get the compound type once for all queries
    object_type = extracted_keywords.get('compound_type', '')
    
    # Process each search result category
    filtered_search_results = {}
    relevance_scores = {}
    
    for category, results in search_results.items():
        filtered_results = []
        category_scores = []
        
        query_index = int(category.split('_')[1]) - 1
        current_query = search_queries[query_index] if query_index < len(search_queries) else "Unknown query"
        
        print(f"\nGrading content for: {current_query}")
        print(f"Object type: {object_type}")
        
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "")
            url = result.get("url", "")
            
            # Truncate content if too long to avoid token limits
            if len(content) > 1500:
                content = content[:1500] + "..."
                
            # Grade the content
            response = web_content_grader.invoke({
                "content": content,
                "search_query": search_queries[query_index],
                "object_type": object_type
            })
            
            # Extract yes/no from response - fixed to use binary_score
            grade = response.binary_score.lower()
            category_scores.append(grade)
            
            print(f"\nSearch Query: {current_query}")
            print(f"Content: {content[:100]}...")
            print(f"Relevance: {grade}")
            
            if grade == "yes":
                filtered_results.append(result)
                
        # Add to filtered results if any relevant content was found
        if filtered_results:    
            filtered_search_results[category] = filtered_results
        relevance_scores[category] = category_scores
            
    # Print Summary
    print("\nGrading Summary:")
    for category, scores in relevance_scores.items():
        query_index = int(category.split('_')[1]) - 1
        current_query = search_queries[query_index] if query_index < len(search_queries) else "Unknown query"
        relevant_count = scores.count("yes")
        total_count = len(scores)
        print(f"\nQuery: {current_query}")
        print(f"Relevant Results: {relevant_count}/{total_count}")
        
    # Add detailed information to debug_info
    updated_debug_info = {
        **debug_info,
        "stage": "content_grading",
        "original_search_results": sum(len(results) for results in search_results.values()),
        "filtered_search_results": sum(len(results) for results in filtered_search_results.values()),
        "relevance_scores": relevance_scores
    }
    
    return {
        "filtered_search_results": filtered_search_results,
        "debug_info": updated_debug_info
    }   

def create_step_back_analyzer(llm):
    """Create a step_back_analysis function with a specific LLM"""
    def step_back_analysis(state: State) -> Dict:
        """Perform step-back analysis using filtered search results and extracted keywords"""
        input_text = state.get("input_text", "")
        extracted_keywords = state.get("extracted_keywords", {})
        filtered_search_results = state.get("filtered_search_results", {})
        debug_info = state.get("debug_info", {})
        
        print("="*25,"Performing Step-Back Analysis","="*25)
        print()
    
        conversation_logger = ConversationLogger()
        
        # Initialize step-back analyzer
        analyzer = StepBackAnalyzer(llm, conversation_logger)
        
        # Prepare context from filtered web content
        context = ""
        for category, results in filtered_search_results.items():
            for result in results:
                title = result.get('title', 'No title')
                content = result.get('content', '')
                context += f"\nTitle: {title}\nContent: {content}\n---\n"
        
        # Augment input text with web context if available
        augmented_description = input_text
        if context:
            augmented_description += f"\n\nAdditional context from research:\n{context}"
        
        # Perform step-back analysis
        analysis_result = analyzer.perform_analysis(
            query=input_text,
            description=augmented_description,
            keyword_data=extracted_keywords
        )
        
        # Create updated state
        updated_state = {
            "step_back_analysis": analysis_result,
            "debug_info": {
                **debug_info,
                "stage": "step_back_analysis",
                "analysis_complete": analysis_result is not None
            }
        }
        
        print("\nStep-back analysis complete.")
        return updated_state
        
    return step_back_analysis   

def hallucination_checker(state: State) -> Dict:
    """Check for hallucinations in the step-back analysis"""
    step_back_analysis = state.get("step_back_analysis", {})
    filtered_search_results = state.get("filtered_search_results", {})
    debug_info = state.get("debug_info", {})
    
    print("="*25,"Checking for Hallucinations","="*25)
    print()
    
    hallucination_llm = ChatOllama(
        model="llama3.2:3b-instruct-q4_K_M",
        temperature=0.0,
        base_url="http://localhost:11434"
    )
    
    structured_hallucination_checker = hallucination_llm.with_structured_output(GradeHallucinations)
    
    # Gather all facts from search results
    facts = ""
    for category, results in filtered_search_results.items():
        for result in results:
            title = result.get('title', 'No title')
            content = result.get('content', '')
            facts += f"\nTitle: {title}\nContent: {content}\n---\n"
            
    # Extract analysis content
    analysis_content = ""
    if isinstance(step_back_analysis, dict):
        # Try to access common fields that might contain the analysis
        if 'principles' in step_back_analysis:
            principles = step_back_analysis.get('principles', [])
            analysis_content += "Core Principles:\n" + "\n".join([f"- {p}" for p in principles]) + "\n\n"
            
        if 'abstractions' in step_back_analysis:
            abstractions = step_back_analysis.get('abstractions', [])
            analysis_content += "Shape Components:\n" + "\n".join([f"- {a}" for a in abstractions]) + "\n\n"
            
        if 'approach' in step_back_analysis:
            approach = step_back_analysis.get('approach', [])
            analysis_content += "Implementation Steps:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(approach)]) + "\n"
        
        # If we still don't have content, try the 'analysis' field
        if not analysis_content and 'analysis' in step_back_analysis:
            analysis_content = step_back_analysis['analysis']
    else:
        # If it's not a dict, convert to string
        analysis_content = str(step_back_analysis)
    
    print("Checking for hallucinations in the step back analysis")
    print(f"Number of fact sources: {sum(len(results) for results in filtered_search_results.values())}")
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", STEP_BACK_HALLUCINATION_CHECKER_SYSTEM_PROMPT),
        ("human", "Set of facts: {facts}\n\nStep back analysis: {analysis}\n\nDoes this analysis contain any hallucinations or made-up information not supported by the facts? Answer with ONLY 'yes' if there are hallucinations, or ONLY 'no' if the analysis is accurate and supported by facts. Then provide a brief explanation.")
    ])
    
    hallucination_checker = hallucination_prompt | structured_hallucination_checker
    
    response = hallucination_checker.invoke({
        "facts": facts,
        "analysis": analysis_content
    })
    
    hallucination_free = response.binary_score.lower() == "yes"
    
    print(f"\nHallucination check result: {'Grounded in facts' if hallucination_free else 'Contains hallucinations'}")
    
    updated_state = {
        "hallucination_check": {
            "is_hallucination_free": hallucination_free,
            "result": response.binary_score
        },
        "debug_info": {
            **debug_info,
            "stage": "hallucination_check",
            "hallucination_free": hallucination_free
        }
    }
            
    return updated_state


def step_back_hallucination_router(state: State):
    """Route to appropriate next step based on hallucination check results"""
    hallucination_check = state.get("hallucination_check", {})
    is_hallucination_free = hallucination_check.get("is_hallucination_free", False)
    
    print(f"Hallucination router decision: {'Proceed to grading' if is_hallucination_free else 'Redo step back analysis'}")
    
    if is_hallucination_free:
        return "grade_step_back_analysis"
    else:
        return "run_step_back_analysis"

def step_back_analysis_grader(state: State) -> Dict:
    """Grade the step-back analysis"""
    step_back_analysis = state.get("step_back_analysis", {})
    input_text = state.get("input_text", "")
    extracted_keywords = state.get("extracted_keywords", {})
    debug_info = state.get("debug_info", {})
    
    print("="*25,"Grading Step-Back Analysis","="*25)
    print()
    
    step_back_analysis_grader_llm = ChatOllama(
        model="llama3.2:3b-instruct-q4_K_M",
        temperature=0.0,
        base_url="http://localhost:11434"
    )
    
    structured_step_back_analysis_grader = step_back_analysis_grader_llm.with_structured_output(GradeStepBackAnalysis)
    
    # Extract analysis content
    analysis_content = ""
    if isinstance(step_back_analysis, dict):
        # Try to access common fields that might contain the analysis
        if 'principles' in step_back_analysis:
            principles = step_back_analysis.get('principles', [])
            analysis_content += "Core Principles:\n" + "\n".join([f"- {p}" for p in principles]) + "\n\n"
            
        if 'abstractions' in step_back_analysis:
            abstractions = step_back_analysis.get('abstractions', [])
            analysis_content += "Shape Components:\n" + "\n".join([f"- {a}" for a in abstractions]) + "\n\n"
            
        if 'approach' in step_back_analysis:
            approach = step_back_analysis.get('approach', [])
            analysis_content += "Implementation Steps:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(approach)]) + "\n"
        
        # If we still don't have content, try the 'analysis' field
        if not analysis_content and 'analysis' in step_back_analysis:
            analysis_content = step_back_analysis['analysis']
    else:
        # If it's not a dict, convert to string
        analysis_content = str(step_back_analysis)
    
    # Extract keyword information
    core_type = extracted_keywords.get('core_type', '')
    compound_type = extracted_keywords.get('compound_type', '')
    modifiers = extracted_keywords.get('modifiers', [])
    
    # Format object description
    object_desc = compound_type if compound_type else f"{' '.join(modifiers)} {core_type}".strip()
    
    print(f"Grading step back analysis for {object_desc}")
    
    step_back_analysis_grader_prompt = ChatPromptTemplate.from_messages([
        ("system", STEP_BACK_GRADER_PROMPT),
        ("human", "Original query: {query}\n\nObject to model: {object_desc}\n\nStep back analysis: {analysis}\n\nRate this analysis on a scale of 0-10 (where 0 is completely inadequate and 10 is excellent) for 3D modeling purposes. Provide detailed feedback on strengths and areas for improvement.")
    ])
    
    step_back_analysis_grader = step_back_analysis_grader_prompt | structured_step_back_analysis_grader
    
    response = step_back_analysis_grader.invoke({
        "query": input_text,
        "object_desc": object_desc,
        "analysis": analysis_content
    })
    
    rating = response.rating
    feedback = response.feedback
    
    print(f"Grader: Rating: {rating}")
    
    # Check if rating meets threshold (7)
    is_good_quality = rating >= 7
    
    print(f"\nAnalysis quality rating: {rating}/10 ({'Good' if is_good_quality else 'Needs improvement'})")
    print(f"Feedback: {feedback}")
    print(f"Grader: Is good quality: {is_good_quality}")
    
    # Create updated state while preserving existing state
    updated_state = {
        **state,  # Preserve ALL existing state
        "analysis_grade": {
            "is_good_quality": is_good_quality,
            "rating": rating,
            "feedback": feedback
        },
        "debug_info": {
            **debug_info,
            "stage": "step_back_analysis_grader",
            "rating": rating,
            "good_quality": is_good_quality,
            "feedback": feedback
        }
    }
    
    print(f"DEBUG - Updated state keys: {list(updated_state)}")
    
    return updated_state

def step_back_quality_router(state: State):
    """Route to appropriate next step based on analysis quality assessment"""
    analysis_grade = state.get("analysis_grade", {})
    is_good_quality = analysis_grade.get("is_good_quality", False)
    rating = analysis_grade.get("rating", 0)
    
    # Add a retry counter to the state
    retry_count = state.get("debug_info", {}).get("step_back_retry_count", 0)
    
    # If we've tried too many times, move forward anyway
    if retry_count >= 3:
        print(f"Maximum step-back retry count reached, proceeding despite rating: {rating}/10")
        return "analyze_query"
    
    if is_good_quality:
        return "analyze_query"
    else:
        # Update the retry counter in the state
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["step_back_retry_count"] = retry_count + 1
        return "run_step_back_analysis"
    
# In graph_state_tools.py

# Create a function to prepare query analysis data
def prepare_query_analysis_data(state: State) -> Dict:
    """Extract and format data needed for query analysis."""
    input_text = state.get("input_text", "")
    extracted_keywords = state.get("extracted_keywords", {})
    step_back_analysis = state.get("step_back_analysis", {})
    filtered_search_results = state.get("filtered_search_results", {})
    
    # Format step back analysis content
    step_back_content = ""
    if step_back_analysis and isinstance(step_back_analysis, dict):
        # Format principles
        principles = step_back_analysis.get('principles', [])
        if principles:
            step_back_content += "Design Principles:\n" + "\n".join([f"- {p}" for p in principles]) + "\n\n"
        
        # Format abstractions/components
        abstractions = step_back_analysis.get('abstractions', [])
        if abstractions:
            step_back_content += "Components:\n" + "\n".join([f"- {a}" for a in abstractions]) + "\n\n"
        
        # Format approach/steps
        approach = step_back_analysis.get('approach', [])
        if approach:
            step_back_content += "Implementation Steps:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(approach)]) + "\n"
        
        # If we still don't have content, try the 'analysis' field
        if not step_back_content and 'analysis' in step_back_analysis:
            step_back_content = step_back_analysis['analysis']
    elif step_back_analysis:  # Handle case where it's a string or other type
        step_back_content = str(step_back_analysis)
    
    # Format keyword information
    keyword_info = ""
    if extracted_keywords:
        core_type = extracted_keywords.get('core_type', '')
        compound_type = extracted_keywords.get('compound_type', '')
        modifiers = extracted_keywords.get('modifiers', [])
        
        keyword_info = f"""
        Core Type: {core_type}
        Modifiers: {', '.join(modifiers)}
        Compound Type: {compound_type}
        """
    
    # Extract relevant information from search results
    search_content = ""
    try:
        if filtered_search_results:
            results_summary = []
            for category, results in filtered_search_results.items():
                for result in results:
                    title = result.get('title', 'No title')
                    # Extract just a snippet from the content for relevance
                    content = result.get('content', '')
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    results_summary.append(f"Title: {title}\nSnippet: {snippet}")
            
            if results_summary:
                search_content = "Search Results Highlights:\n" + "\n\n".join(results_summary[:2])
    except Exception as e:
        print(f"Error processing search results: {e}")
        search_content = "Error processing search results"
    
    return {
        "input_text": input_text,
        "keyword_info": keyword_info,
        "step_back_content": step_back_content,
        "search_content": search_content
    }

# Create a function to perform the query analysis with LLM
def perform_query_analysis(analysis_data: Dict, query_analysis_llm) -> Dict:
    """Perform the actual query analysis using LLM."""
    query_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_ANALYSIS_PROMPT),
        ("human", """
         Original query: {input_text}
         
         Extracted Keywords: {keyword_info}
         
         Step-back Analysis: {step_back_content}
         
         Search Results: {search_content}
         
         Based on this information, analyze how to best find similar SCAD code examples in our database.
         Your response should include search strategy, enhanced query, important attributes, style preference, complexity, and code similarities.
         Also identify likely OpenSCAD techniques that will be needed for implementation.
         """)
    ])
    
    structured_query_analysis = query_analysis_llm.with_structured_output(SCADQueryAnalysis)
    query_analyzer = query_analysis_prompt | structured_query_analysis
    
    print(f"Analyzing query: '{analysis_data['input_text']}'")
    try:
        result = query_analyzer.invoke(analysis_data)
        return result
    except Exception as e:
        print(f"Error in query analysis: {e}")
        # Return a default SCADQueryAnalysis object if parsing fails
        return SCADQueryAnalysis(
            search_strategy="hybrid",
            enhanced_query=analysis_data['input_text'],
            important_attributes=[],
            style_preference="Parametric",
            complexity="MEDIUM",
            code_similarities=[]
        )

# Create a function to identify likely OpenSCAD techniques
def identify_likely_techniques(result, step_back_content: str) -> List[str]:
    """Identify likely OpenSCAD techniques based on analysis."""
    likely_techniques = []
    
    # Look for common OpenSCAD operations in the analysis
    technique_keywords = {
        "union": ["combine", "join", "merge", "add", "union"],
        "difference": ["subtract", "cut", "remove", "hollow", "difference"],
        "intersection": ["intersect", "overlap", "common", "shared"],
        "translate": ["move", "position", "place", "translate", "shift"],
        "rotate": ["turn", "spin", "rotation", "angle", "rotate"],
        "scale": ["resize", "proportion", "enlarge", "reduce", "scale"],
        "mirror": ["reflect", "flip", "mirror", "symmetry"],
        "hull": ["envelope", "surround", "outline", "hull", "convex"],
        "minkowski": ["smooth", "round", "bevel", "sum", "minkowski"],
        "extrude": ["extend", "extrude", "raise", "linear", "path"],
        "pattern": ["repeat", "array", "grid", "pattern", "duplicate"]
    }
    
    # Check if result is None or doesn't have the expected attributes
    has_important_attributes = result is not None and hasattr(result, 'important_attributes') and result.important_attributes is not None
    has_code_similarities = result is not None and hasattr(result, 'code_similarities') and result.code_similarities is not None
    
    # Check step back analysis for technique keywords
    for technique, keywords in technique_keywords.items():
        # Check if any of the keywords for this technique appear in the analysis
        for keyword in keywords:
            keyword_in_attributes = has_important_attributes and any(keyword.lower() in attr.lower() for attr in result.important_attributes)
            keyword_in_similarities = has_code_similarities and any(keyword.lower() in sim.lower() for sim in result.code_similarities)
            
            if (step_back_content and keyword.lower() in step_back_content.lower()) or \
               keyword_in_attributes or \
               keyword_in_similarities:
                likely_techniques.append(technique)
                break
    
    return list(set(likely_techniques))  # Remove duplicates

# Create a function to format the final analysis output
def format_query_analysis(result, likely_techniques: List[str]) -> Dict:
    """Format the final query analysis output."""
    # Fallback if result is None
    if result is None:
        return {
            "search_strategy": "hybrid",
            "enhanced_query": "guitar 3D model",  # Provide a reasonable default
            "important_attributes": [],
            "style_preference": "Parametric",
            "complexity": "MEDIUM",
            "similarities_to_check": [],
            "techniques_needed": likely_techniques or ["union", "difference"]
        }
    
    return {
        "search_strategy": result.search_strategy,
        "enhanced_query": result.enhanced_query,
        "important_attributes": result.important_attributes,
        "style_preference": result.style_preference,
        "complexity": result.complexity,
        "similarities_to_check": result.code_similarities,
        "techniques_needed": likely_techniques
    }

# Step 5: Finally, update the main analyze_query function to use these modular functions
def analyze_query(state: State) -> Dict:
    """Analyze the user query to determine the best approach for vectorstore search."""
    debug_info = state.get("debug_info", {})
    input_text = state.get("input_text", "")
    
    print("="*25,"Performing Query Analysis for SCAD Code Retrieval","="*25)
    print()
    
    query_analysis_llm = ChatOllama(
        model="llama3.2:3b-instruct-q4_K_M",
        temperature=0.0,
        base_url="http://localhost:11434"
    )
    
    try:
        # Use the modular functions
        analysis_data = prepare_query_analysis_data(state)
        result = perform_query_analysis(analysis_data, query_analysis_llm)
        step_back_content = analysis_data.get("step_back_content", "")
        likely_techniques = identify_likely_techniques(result, step_back_content)
        query_analysis = format_query_analysis(result, likely_techniques)
        
        # Print analysis results
        print("\nSCAD Code Retrieval Analysis:")
        print(f"Search Strategy: {query_analysis['search_strategy']}")
        print(f"Enhanced Query: {query_analysis['enhanced_query']}")
        print(f"Important Attributes: {query_analysis['important_attributes']}")
        print(f"Style Preference: {query_analysis['style_preference']}")
        print(f"Complexity: {query_analysis['complexity']}")
        print(f"Code Similarities: {query_analysis['similarities_to_check']}")
        print(f"Likely Techniques: {query_analysis['techniques_needed']}")
        
    except Exception as e:
        # If any part of the process fails, create a fallback analysis
        print(f"Error in query analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Create default query analysis
        query_analysis = {
            "search_strategy": "hybrid",
            "enhanced_query": input_text + " 3D model",  # Add "3D model" to the original query
            "important_attributes": [],
            "style_preference": "Parametric",
            "complexity": "MEDIUM",
            "similarities_to_check": [],
            "techniques_needed": ["union", "difference", "translate"] # Basic techniques
        }
        
        print("\nUsing fallback query analysis due to error")
        print(f"Search Strategy: {query_analysis['search_strategy']}")
        print(f"Enhanced Query: {query_analysis['enhanced_query']}")
    
    updated_state = {
        **state,
        "query_analysis": query_analysis,
        "debug_info": {
            **debug_info,
            "stage": "query_analysis",
            "query_analysis": query_analysis
        }
    }
    
    return updated_state

def create_examples_retriever(kb_instance):
    """Create an examples retriever function with the provided LLM and knowledge base"""
    
    def retrieve_similar_examples(state: State) -> Dict:
        """Retrieve similar SCAD code examples from the knowledge base based on query analysis"""
        input_text = state.get("input_text", "")
        extracted_keywords = state.get("extracted_keywords", {})
        step_back_analysis = state.get("step_back_analysis", {})
        query_analysis = state.get("query_analysis", {})
        debug_info = state.get("debug_info", {})
        
        print("="*25,"Retrieving Similar SCAD Code Examples","="*25)
        print()
        
        # Access the knowledge base
        knowledge_base = kb_instance
        
        # Default values if query_analysis is missing
        search_strategy = query_analysis.get('search_strategy', 'hybrid')
        enhanced_query = query_analysis.get('enhanced_query', input_text)
        important_attributes = query_analysis.get('important_attributes', [])
        style_preference = query_analysis.get('style_preference', 'Parametric')
        complexity = query_analysis.get('complexity', 'MEDIUM')
        code_similarities = query_analysis.get('similarities_to_check', [])
        techniques_needed = query_analysis.get('techniques_needed', [])
        
        # Set similarity threshold based on search strategy
        if search_strategy == 'semantic':
            similarity_threshold = 0.55  # Lower threshold for semantic search
        elif search_strategy == 'keyword':
            similarity_threshold = 0.65  # Higher threshold for keyword search
        else:  # hybrid
            similarity_threshold = 0.6   # Balanced threshold for hybrid search
        
        print(f"Search Strategy: {search_strategy}")
        print(f"Enhanced Query: {enhanced_query}")
        print(f"Similarity Threshold: {similarity_threshold}")
        print(f"Techniques Needed: {techniques_needed}")
        
        # Augment the step back analysis with code similarities and techniques
        if isinstance(step_back_analysis, dict):
            # Create a copy to avoid modifying the original
            augmented_step_back = dict(step_back_analysis)
            
            # Add code similarities to the step back analysis
            if 'principles' not in augmented_step_back or not augmented_step_back['principles']:
                augmented_step_back['principles'] = []
            
            # Add code-related principles
            if code_similarities:
                for similarity in code_similarities:
                    if similarity not in augmented_step_back['principles']:
                        augmented_step_back['principles'].append(f"Code Structure: {similarity}")
            
            # Add techniques needed to the step back analysis
            if techniques_needed:
                for technique in techniques_needed:
                    if technique not in augmented_step_back['principles']:
                        augmented_step_back['principles'].append(f"Implementation Technique: {technique}")
        else:
            augmented_step_back = step_back_analysis
        
        # Retrieve examples using the knowledge base's get_examples method
        examples_result, metadata = knowledge_base.get_examples(
            description=enhanced_query,
            step_back_result=augmented_step_back,
            keyword_data=extracted_keywords, 
            similarity_threshold=similarity_threshold,
            return_metadata=True,
            max_results=5  # Limit to top 5 most relevant examples
        )
        
        # Check if we got results
        if not examples_result:
            print("\nNo similar SCAD code examples found in knowledge base.")
            similar_examples = []
        else:
            # Process the examples
            similar_examples = []
            print(f"\nFound {len(examples_result)} similar SCAD code examples:")
                
            for i, example in enumerate(examples_result, 1):
                example_data = example['example']
                    
                # Extract example details
                example_info = {
                    'id': example_data.get('id', f"example_{i}"),
                    'description': example_data.get('document', example_data.get('metadata', {}).get('description', 'No description')),
                    'code': example_data.get('metadata', {}).get('code', 'No code available'),
                    'score': example.get('score', 0),
                    'metadata': example_data.get('metadata', {}),
                    'score_details': example.get('score_breakdown', {})
                }
                
                # Extract additional metadata for rich preview
                metadata = example_data.get('metadata', {})
                if metadata:
                    # Extract techniques used
                    techniques_used = metadata.get('techniques_used', [])
                    if techniques_used:
                        example_info['techniques_used'] = techniques_used
                    
                    # Extract code metrics
                    code_metrics = metadata.get('code_metrics', {})
                    if code_metrics:
                        example_info['code_metrics'] = code_metrics
                    
                    # Extract primary parameters
                    primary_parameters = metadata.get('primary_parameters', [])
                    if primary_parameters:
                        example_info['primary_parameters'] = primary_parameters
                
                similar_examples.append(example_info)
                    
                # Print enhanced example summary
                print(f"\nExample {i} (Score: {example_info['score']:.3f}):")
                print(f"ID: {example_info['id']}")
                print(f"Description: {example_info['description'][:100]}...")
                    
                # Print score breakdown if available
                if 'component_scores' in example.get('score_breakdown', {}):
                    print("Score Breakdown:")
                    for name, score in example['score_breakdown']['component_scores'].items():
                        print(f"  {name}: {score:.3f}")
                
                # Print techniques used if available
                if 'techniques_used' in example_info:
                    print(f"Techniques Used: {', '.join(example_info['techniques_used'])}")
                
                # Print code metrics if available  
                if 'code_metrics' in example_info:
                    metrics = example_info['code_metrics']
                    print(f"Code Metrics: {metrics.get('line_count', 0)} lines, " +
                          f"{metrics.get('module_count', 0)} modules, " +
                          f"{metrics.get('parameter_count', 0)} parameters")
                        
                # Add code preview
                if example_info['code']:
                    code_preview = example_info['code'].split('\n')[:3]
                    code_preview.append('...')
                    print("Code Preview:")
                    for line in code_preview:
                        print(f"  {line}")
                    
                    # Print top parameters if available
                    if 'primary_parameters' in example_info and example_info['primary_parameters']:
                        print("Top Parameters:")
                        for param in example_info['primary_parameters'][:3]:  # Show only top 3
                            name = param.get('name', '')
                            value = param.get('value', '')
                            comment = param.get('comment', '')
                            if comment:
                                print(f"  {name} = {value} // {comment}")
                            else:
                                print(f"  {name} = {value}")
        
        # Post-processing: Sort examples by relevance to needed techniques if we have both
        if techniques_needed and similar_examples:
            # Define a scoring function based on technique match
            def technique_relevance_score(example):
                base_score = example['score']
                technique_bonus = 0
                
                # Get techniques used in this example
                example_techniques = example.get('techniques_used', [])
                
                # Calculate bonus for matching techniques
                if example_techniques and techniques_needed:
                    matches = sum(1 for tech in techniques_needed if tech in example_techniques)
                    if matches > 0:
                        # Give up to 0.2 bonus based on the percentage of matches
                        technique_bonus = 0.2 * (matches / len(techniques_needed))
                
                return base_score + technique_bonus
            
            # Sort examples by the combined score
            similar_examples.sort(key=technique_relevance_score, reverse=True)
            print("\nExamples re-ranked based on technique relevance.")
        
        # Create updated state including the original state
        updated_state = {
            **state,  # Preserve ALL existing state
            "similar_examples": similar_examples,
            "retrieved_metadata": metadata,
            "debug_info": {
                **debug_info,
                "stage": "example_retrieval",
                "similar_examples_count": len(similar_examples),
                "query_metadata": metadata,
                "used_enhanced_query": enhanced_query,
                "techniques_needed": techniques_needed,
                "search_strategy": search_strategy,
                "similarity_threshold": similarity_threshold
            }
        }
        
        print(f"\nExample retrieval complete. Found {len(similar_examples)} SCAD code examples.")
        return updated_state
    
    return retrieve_similar_examples

def create_scad_code_generator(llm, kb_instance):
    """Create a SCAD code generator function with a specific LLM and knowledge base"""
    from prompts import BASIC_KNOWLEDGE, OPENSCAD_GNERATOR_PROMPT_TEMPLATE
    from scad_code_validator import validate_scad_code, validate_scad_syntax
    from parameter_tuner import ParameterTuner, identify_parameters_from_examples, get_user_parameter_input, suggest_parameters_from_description
    
    # Define helper functions within the closure
    def extract_techniques(code):
        """Extract OpenSCAD techniques used in the code"""
        techniques = []
        
        # Check for common OpenSCAD operations
        operations = {
            "union": r"union\s*\(",
            "difference": r"difference\s*\(",
            "intersection": r"intersection\s*\(",
            "translate": r"translate\s*\(",
            "rotate": r"rotate\s*\(",
            "scale": r"scale\s*\(",
            "mirror": r"mirror\s*\(",
            "hull": r"hull\s*\(",
            "minkowski": r"minkowski\s*\(",
            "linear_extrude": r"linear_extrude\s*\(",
            "rotate_extrude": r"rotate_extrude\s*\("
        }
        
        for technique, pattern in operations.items():
            if re.search(pattern, code):
                techniques.append(technique)
        
        # Check for more complex patterns
        if "for" in code and "(" in code:
            techniques.append("iteration")
        
        if "module" in code:
            techniques.append("modular")
        
        if "function" in code:
            techniques.append("functional")
        
        if re.search(r"if\s*\(", code):
            techniques.append("conditional")
        
        return techniques

    def extract_parameters(code):
        """Extract primary parameters from OpenSCAD code"""
        # Look for parameters defined at the top level
        params = re.findall(r"^\s*([a-zA-Z0-9_]+)\s*=\s*([^;]+);(?:\s*\/\/\s*(.+))?", code, re.MULTILINE)
        
        # Process into a more usable format
        parameters = []
        for name, value, comment in params:
            # Skip internal variables
            if name.startswith("_") or "tmp" in name.lower() or "temp" in name.lower():
                continue
            
            parameters.append({
                "name": name.strip(),
                "value": value.strip(),
                "comment": comment.strip() if comment else None
            })
        
        # Sort by position in file (earlier parameters are likely more important)
        return parameters[:10]  # Return just the top 10 parameters
    
    def generate_scad_code(state: State) -> Dict:
        """Generate OpenSCAD code using the prepared inputs from the state"""
        input_text = state.get("input_text", "")
        step_back_analysis = state.get("step_back_analysis", {})
        similar_examples = state.get("similar_examples", [])
        query_analysis = state.get("query_analysis", {})
        filtered_search_results = state.get("filtered_search_results", {})
        retrieved_metadata = state.get("retrieved_metadata", {})
        debug_info = state.get("debug_info", {})
        
        export_model = "n"
        formats = []
        
        print("\n" + "="*50)
        print("GENERATING SCAD CODE")
        print("="*50)
        
        # Print model information
        model_name = getattr(llm, 'model_name', str(llm))
        print(f"Using model: {model_name}")
        
        try:
            # Format step-back analysis
            print("Formatting step-back analysis...")
            step_back_text = ""
            if step_back_analysis:
                # Handle the case where step_back_analysis is None or not a dictionary
                if isinstance(step_back_analysis, dict):
                    principles = step_back_analysis.get('principles', [])
                    abstractions = step_back_analysis.get('abstractions', [])
                    approach = step_back_analysis.get('approach', [])
                    
                    step_back_text = f"""
                    CORE PRINCIPLES:
                    {chr(10).join(f'- {p}' for p in principles)}
                    
                    SHAPE COMPONENTS:
                    {chr(10).join(f'- {a}' for a in abstractions)}
                    
                    IMPLEMENTATION STEPS:
                    {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(approach))}
                    """
                else:
                    # If it's not a dictionary, convert to string and use as is
                    step_back_text = str(step_back_analysis) if step_back_analysis else ""
            
            # Format examples
            print("Formatting examples...")
            examples_text = []
            for example in similar_examples:
                example_text = f"""
                Example ID: {example.get('id', 'unknown')}
                Score: {example.get('score', 0.0):.3f}
                Description: {example.get('description', '')}
                Code:
                ```scad
                {example.get('code', 'No code available')}
                ```
                """
                examples_text.append(example_text)
            
            examples_formatted = "\n".join(examples_text)
            
            # Format the enhanced query if available
            print("Formatting enhanced query...")
            enhanced_query = query_analysis.get('enhanced_query', input_text) if query_analysis else input_text
            
            # Format web content from filtered search results
            print("Formatting web content...")
            web_content = ""
            for category, results in filtered_search_results.items():
                for result in results:
                    title = result.get('title', 'No title')
                    content = result.get('content', '')
                    if content:
                        web_content += f"\nTitle: {title}\nContent: {content}\n---\n"
            
            # If web_content is empty, display a message
            if not web_content.strip():
                web_content = "No relevant web content was found."
            else:
                web_content = "Here is relevant information from web searches:\n" + web_content
            
            # Try to identify parameters from examples and customize them
            parameter_suggestions = ""
            template_suggestion = "Consider using appropriate module structure based on the object type."
            
            # First try to extract parameters
            try:
                # First extract parameters from the description
                print("\nAnalyzing description to identify appropriate parameters...")
                desc_parameters = None
                # Extract information about the object from state
                object_type = ""
                if state.get("extracted_keywords") and "core_type" in state.get("extracted_keywords", {}):
                    object_type = state.get("extracted_keywords", {}).get("core_type", "")
                
                # Get step-back analysis for context
                step_back_result = state.get("step_back_analysis", {})
                
                # Get parameters from description
                desc_parameters = suggest_parameters_from_description(input_text, object_type, step_back_result, llm)
                
                # Then identify parameters from examples
                example_parameters = None
                if similar_examples:
                    print("\nAnalyzing examples for additional parameters...")
                    example_parameters = identify_parameters_from_examples(similar_examples, input_text, llm)
                
                # Merge the parameters, prioritizing those from the description
                merged_parameters = {}
                
                # Add parameters from examples first (if any)
                if example_parameters:
                    for name, info in example_parameters.items():
                        merged_parameters[name] = info
                
                # Add or overwrite with parameters from description (if any)
                if desc_parameters:
                    for name, info in desc_parameters.items():
                        merged_parameters[name] = info
                
                # Get user input for the merged parameters
                parameters = None
                if merged_parameters:
                    print("\nSuggested parameters based on your description and similar examples:")
                    parameters = get_user_parameter_input(merged_parameters)
                    
                    # Format parameter information for the prompt
                    if parameters:
                        parameter_suggestions = "SUGGESTED PARAMETERS:\n"
                        for name, info in parameters.items():
                            value = info.get("value")
                            description = info.get("description", "")
                            
                            # Format based on type
                            if isinstance(value, list):
                                # Vector format
                                value_str = f"[{', '.join(str(x) for x in value)}]"
                            elif isinstance(value, bool):
                                # Boolean format
                                value_str = "true" if value else "false"
                            else:
                                value_str = str(value)
                            
                            parameter_suggestions += f"{name} = {value_str}; // {description}\n"
                else:
                    # If no parameters could be identified, provide some default ones based on object type
                    print("\nNo specific parameters could be identified. Using default parameters for this object type.")
                    parameter_suggestions = """SUGGESTED PARAMETERS:
// Basic dimensions that should work for most objects
width = 100; // Width of the object
height = 150; // Height of the object
depth = 70; // Depth of the object
wall_thickness = 2; // Wall thickness for hollow objects
resolution = 100; // Resolution for curved surfaces ($fn value)
"""
            except Exception as e:
                error_message = str(e)
                print(f"Error identifying parameters: {error_message}")
                import traceback
                traceback.print_exc()
                # Provide default parameters as fallback
                parameter_suggestions = """SUGGESTED PARAMETERS:
// Default parameters since extraction failed
width = 100; // Width of the object
height = 150; // Height of the object
depth = 70; // Depth of the object
wall_thickness = 2; // Wall thickness for hollow objects
resolution = 100; // Resolution for curved surfaces ($fn value)
"""

            # Now try to select a template in a separate try/except block
            try:
                # Select appropriate template based on object characteristics
                print("\nSelecting appropriate template...")
                
                # Import template utilities
                from scad_templates import select_template_for_object, generate_template_params, apply_template, SCAD_TEMPLATES

                # Get object type and modifiers from extracted keywords
                modifiers = []
                extracted_keywords = state.get("extracted_keywords", {})
                
                if extracted_keywords:
                    object_type = extracted_keywords.get("core_type", "")
                    modifiers = extracted_keywords.get("modifiers", [])
                
                # Select appropriate template based on object characteristics
                template_name = select_template_for_object(object_type, modifiers, step_back_result)
                
                # Generate template parameters
                template_params = generate_template_params(object_type, modifiers, step_back_result)
                
                # Apply the template to get example code
                template_code = apply_template(template_name, template_params)
                
                # Create template descriptions to inform the LLM about available templates
                template_descriptions = "AVAILABLE TEMPLATES:\n"
                for template_key, template_value in SCAD_TEMPLATES.items():
                    template_descriptions += f"- {template_key}: For {template_key}-type objects\n"
                
                # Add template information to the inputs
                template_suggestion = f"""
                SUGGESTED TEMPLATE:
                The object appears to be a "{template_name}" type. Here's a suggested structure:
                
                ```scad
                {template_code}
                ```
                
                {template_descriptions}
                
                Feel free to use this template as a starting point and modify it as needed.
                """
                
                print(f"Selected template: {template_name}")
            except Exception as e:
                print(f"Error generating template suggestion: {str(e)}")
                template_suggestion = "Consider using appropriate module structure based on the object type."

            # Prepare the prompt inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples_formatted,
                "request": enhanced_query,
                "step_back_analysis": step_back_text.strip() if step_back_text else "",
                "template_suggestion": template_suggestion,  # Use the dynamically created template suggestion
                "parameter_suggestions": parameter_suggestions,  # Use the dynamically built parameter suggestions
                "web_content": web_content  # Add the formatted web content
            }
            
            # Generate the prompt and log it
            prompt_value = OPENSCAD_GNERATOR_PROMPT_TEMPLATE.format(**inputs)
            
            # Save the full prompt to a file for debugging
            try:
                with open("full_prompts.txt", "a") as f:
                    f.write("\n\n" + "="*80 + "\n")
                    f.write(f"PROMPT FOR: {input_text}\n")
                    f.write("="*80 + "\n\n")
                    f.write(prompt_value)
                    f.write("\n\n" + "="*80 + "\n\n")
                print(f"Full prompt saved to full_prompts.txt for debugging")
            except Exception as e:
                print(f"Warning: Could not save prompt to file: {str(e)}")
                
            # Continue with code generation
            
            # Get response from the LLM
            print("Generating OpenSCAD code...")
            print("Thinking...", end="", flush=True)
            
            # Get streaming response
            content = ""
            for chunk in llm.stream(prompt_value):
                if hasattr(chunk, 'content'):
                    chunk_content = chunk.content
                else:
                    chunk_content = str(chunk)
                content += chunk_content
                print(".", end="", flush=True)
            
            print("\n")  # New line after progress dots
            
            # Try to extract code with different tag variations
            import re
            code_tags = [
                ('<code>', '</code>'),
                ('```scad', '```'),
                ('```openscad', '```'),
                ('```', '```')
            ]
            
            scad_code = None
            
            # First, check for Claude's artifact formats with multiple patterns
            artifact_patterns = [
                r'<antArtifact\s+[^>]*?type="application/vnd\.ant\.code"[^>]*?>(.*?)</antArtifact>',  # Standard pattern
                r'<antArtifact\s+[^>]*?language="scad"[^>]*?>(.*?)</antArtifact>',  # Language-specific pattern
                r'<antArtifact\s+[^>]*?identifier="[^"]*?"[^>]*?>(.*?)</antArtifact>',  # Identifier pattern
                r'<antArtifact[^>]*?>(.*?)</antArtifact>',  # More lenient pattern
                r'<artifact[^>]*?>(.*?)</artifact>',  # Alternative naming
                r'<ant[^>]*?>(.*?)</ant>'  # Shortest possible match
            ]
            
            # Try each pattern in order
            for pattern in artifact_patterns:
                artifact_match = re.search(pattern, content, re.DOTALL)
                if artifact_match:
                    extracted_code = artifact_match.group(1).strip()
                    # Only use it if we got something substantial
                    if len(extracted_code) > 10:
                        scad_code = extracted_code
                        print(f"\nCode extracted using artifact pattern: {pattern}")
                        break
                    
            # If no artifact found, try standard code tag extraction
            if not scad_code:
                for start_tag, end_tag in code_tags:
                    if start_tag in content and end_tag in content:
                        code_start = content.find(start_tag) + len(start_tag)
                        code_end = content.find(end_tag, code_start)
                        if code_end > code_start:
                            scad_code = content[code_start:code_end].strip()
                            print(f"\nCode extracted using tags: {start_tag}...{end_tag}")
                            break
                        
            # If standard extraction fails, try to fix the raw response
            if not scad_code:
                print("\nAttempting to fix and extract code from response...")
                fixed_content = attempt_to_fix_raw_response(content)
                
                # If the content was modified, try extraction again
                if fixed_content != content:
                    print("Response was fixed, trying to extract code again")
                    
                    # Try extraction with the fixed content
                    for start_tag, end_tag in code_tags:
                        if start_tag in fixed_content and end_tag in fixed_content:
                            code_start = fixed_content.find(start_tag) + len(start_tag)
                            code_end = fixed_content.find(end_tag, code_start)
                            if code_end > code_start:
                                scad_code = fixed_content[code_start:code_end].strip()
                                print("Successfully extracted code from fixed response")
                                break
                                
            # If fixed response extraction fails, check if it looks like complete OpenSCAD code
            if not scad_code and looks_like_complete_openscad(content):
                print("\nThe entire response appears to be OpenSCAD code without tags")
                scad_code = content.strip()
                            
            # If all extraction methods fail, try one more time with a more explicit prompt
            if not scad_code:
                print("\nFirst attempt failed. Retrying with more explicit prompt...")
                retry_prompt = f"""
                            I need ONLY OpenSCAD code for: {input_text}

                            Here are the key elements from the step-back analysis:
                            {step_back_result.get('principles', [''])[0] if isinstance(step_back_result, dict) and 'principles' in step_back_result else ''}

                            DO NOT include any explanations, questions, or text outside the code block.
                            ONLY respond with code inside ```scad tags like this:

                            ```scad
                            // Your OpenSCAD code here
                            ```

                            The code must include basic OpenSCAD elements (cube, sphere, cylinder, etc.) and be syntactically correct.
                            """
            
                # Get streaming response for retry
                retry_content = ""
                print("Retrying...", end="", flush=True)
                for chunk in llm.stream(prompt_value):
                    if hasattr(chunk, 'content'):
                        chunk_content = chunk.content
                    else:
                        chunk_content = str(chunk)
                    retry_content += chunk_content
                    print(".", end="", flush=True)
                
                print("\n")  # New line after progress dots
                
                # Try to extract code from retry attempt
                for start_tag, end_tag in code_tags:
                    if start_tag in retry_content and end_tag in retry_content:
                        code_start = retry_content.find(start_tag) + len(start_tag)
                        code_end = retry_content.find(end_tag, code_start)
                        if code_end > code_start:
                            scad_code = retry_content[code_start:code_end].strip()
                            print("Successfully extracted code from retry attempt")
                            break
                            
                # If retry still failed, try pattern matching on the retry content
                if not scad_code:
                    print("No code tags found in retry response. Attempting pattern matching.")
                    
                    # Try with fixed response and pattern matching
                    fixed_retry = attempt_to_fix_raw_response(retry_content)
                    if fixed_retry != retry_content:
                        # Try extraction with the fixed retry content
                        for start_tag, end_tag in code_tags:
                            if start_tag in fixed_retry and end_tag in fixed_retry:
                                code_start = fixed_retry.find(start_tag) + len(start_tag)
                                code_end = fixed_retry.find(end_tag, code_start)
                                if code_end > code_start:
                                    scad_code = fixed_retry[code_start:code_end].strip()
                                    print("Successfully extracted code from fixed retry response")
                                    break
                                    
                    # If fixing failed, check if response looks like complete OpenSCAD code                
                    if not scad_code and looks_like_complete_openscad(retry_content):
                        print("The retry response appears to be OpenSCAD code without tags")
                        scad_code = retry_content.strip()
            
            # Validate the generated code
            print("\nValidating OpenSCAD code...")
            is_valid_syntax = validate_scad_syntax(scad_code)
            valid_code, validation_messages = validate_scad_code(scad_code)
            
            if not is_valid_syntax:
                print("\nWarning: OpenSCAD code has syntax issues that may prevent proper rendering.")
                # Try to fix simple syntax issues
                fixed_code = scad_code
                # Basic fixes for missing semicolons, braces and parentheses
                if fixed_code.count('{') > fixed_code.count('}'): 
                    fixed_code += '\n' + ('}'*(fixed_code.count('{')-fixed_code.count('}')))
                    print("- Added missing closing braces.")
                if fixed_code.count('(') > fixed_code.count(')'): 
                    fixed_code += ')' * (fixed_code.count('(')-fixed_code.count(')'))
                    print("- Added missing closing parentheses.")
                # Only use the fixed code if we made changes
                if fixed_code != scad_code:
                    scad_code = fixed_code
                    print("- Applied automatic fixes to the code.")
            
            # Print validation results
            print("\nCode validation results:")
            for msg in validation_messages:
                if "Error:" in msg:
                    print(f"❌ {msg}")
                elif "Warning:" in msg:
                    print(f"⚠️ {msg}")
                else:
                    print(f"✓ {msg}")
            
            # Save the generated code to a file
            try:
                with open("output.scad", "w") as f:
                    f.write(scad_code)
                print("\nOpenSCAD code has been generated and saved to 'output.scad'")
            except Exception as e:
                print(f"\nWarning: Could not save code to file: {str(e)}")
            
            # Print a more comprehensive code preview
            print("\nGenerated Code Preview:")
            print("-" * 40)
            lines = scad_code.split('\n')
            total_lines = len(lines)
            
            # Display first 5 lines
            print("// --- Beginning of code ---")
            for i in range(min(5, total_lines)):
                print(lines[i])
                
            # Display middle section if code is long enough
            if total_lines > 15:
                middle_start = max(5, total_lines // 2 - 3)
                print("\n// --- Middle section ---")
                for i in range(middle_start, min(middle_start + 5, total_lines)):
                    print(lines[i])
            
            # Display last 5 lines
            if total_lines > 10:
                print("\n// --- End of code ---")
                for i in range(max(total_lines - 5, 0), total_lines):
                    print(lines[i])
            
            print(f"\n// Total: {total_lines} lines of OpenSCAD code")
            print("-" * 40)
            
            # Ask user if they want to preview the model
            preview_model = input("\nWould you like to generate a preview of this model? (y/n): ").lower().strip()
            
            if preview_model == 'y':
                try:
                    # Import ModelExporter if not already imported
                    from model_exporter import ModelExporter
                    
                    # Get custom preview settings
                    preview_settings = get_preview_settings()
                    
                    print("\nGenerating preview...")
                    preview_path = generate_model_preview(
                        scad_code=scad_code,
                        filename="preview",
                        resolution=preview_settings["resolution"],
                        camera_params=preview_settings["camera_params"],
                        width=preview_settings["width"],
                        height=preview_settings["height"]
                    )
                    
                    if preview_path and os.path.exists(preview_path):
                        print(f"\nPreview generated successfully: {preview_path}")
                        
                        # Try to open the preview image with the default viewer
                        try:
                            import subprocess
                            if os.name == 'posix':  # macOS or Linux
                                if os.path.exists('/usr/bin/open'):  # macOS
                                    subprocess.run(["open", preview_path])
                                else:  # Linux
                                    subprocess.run(["xdg-open", preview_path])
                            elif os.name == 'nt':  # Windows
                                os.startfile(preview_path)
                            print("\nOpened preview in default image viewer.")
                        except Exception as e:
                            print(f"\nCould not automatically open preview: {e}")
                            print(f"Please open the preview image located at: {preview_path}")
                    else:
                        print("\nFailed to generate preview. Please check the logs for details.")
                except Exception as e:
                    print(f"\nError generating preview: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Ask user if they want to tune parameters
            tune_params = input("\nWould you like to tune the parameters for this model? (y/n): ").lower().strip()
            
            if tune_params == 'y':
                try:
                    print("\nTuning parameters...")
                    parameter_tuner = ParameterTuner(llm=llm)
                    tuning_result = parameter_tuner.tune_parameters(scad_code, input_text)
                    
                    if tuning_result.get("success") and tuning_result.get("adjustments"):
                        scad_code = tuning_result.get("updated_code")
                        print("\nParameters tuned successfully!")
                        
                        # Save updated code to file
                        with open("output.scad", "w") as f:
                            f.write(scad_code)
                        
                        print("\nUpdated OpenSCAD code has been saved to 'output.scad'")
                    else:
                        print("\nNo parameter changes were applied.")
                except Exception as e:
                    print(f"\nError tuning parameters: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Ask user if they want to add this to the knowledge base
            add_to_kb = input("\nWould you like to add this example to the knowledge base? (y/n): ").lower().strip()
            
            # Prepare the result object
            result = {
                "success": True,
                "code": scad_code,
                "prompt": prompt_value,
                "add_to_kb": add_to_kb == 'y',
                "validation": {
                    "is_valid_syntax": is_valid_syntax,
                    "is_valid_code": valid_code,
                    "messages": validation_messages
                }
            }
            
            # Only add to knowledge base if requested
            if add_to_kb == 'y' and kb_instance:
                try:
                    # Generate enhanced metadata about the code
                    enhanced_metadata = {
                        **retrieved_metadata,  # Keep existing metadata
                        "code_metrics": {
                            "line_count": len(scad_code.split('\n')),
                            "module_count": scad_code.count("module "),
                            "parameter_count": len(re.findall(r"^\s*([a-zA-Z0-9_]+)\s*=", scad_code, re.MULTILINE))
                        },
                        "techniques_used": extract_techniques(scad_code),
                        "primary_parameters": extract_parameters(scad_code),
                    }
                    
                    # Add step_back_info if step_back_analysis is available and is a dictionary
                    if step_back_analysis and isinstance(step_back_analysis, dict):
                        enhanced_metadata["step_back_info"] = {
                            "principles": step_back_analysis.get("principles", []),
                            "abstractions": step_back_analysis.get("abstractions", []),
                            "approach": step_back_analysis.get("approach", [])
                        }
                    
                    # Call with enhanced metadata (INSIDE if block)
                    kb_success = kb_instance.add_example(input_text, scad_code, metadata=enhanced_metadata)
                    
                    if kb_success:
                        print("Example added to knowledge base!")
                    else:
                        print("Failed to add example to knowledge base.")
                except Exception as e:
                    print(f"Error adding to knowledge base: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            export_model = input("\nWould you like to export this model to other formats? (y/n): ").lower().strip()

            if export_model == 'y':
                # Import the model exporter
                from model_exporter import ModelExporter
                from models import ExportResult
                
                # Create exporter instance
                exporter = ModelExporter(output_dir="output")
                
                print("\nAvailable export formats: stl, off, amf, 3mf, csg, dxf, svg, png")
                formats_input = input("Enter comma-separated formats to export (default: stl): ").strip()
                formats = [f.strip() for f in formats_input.split(',')] if formats_input else ["stl"]
                
                resolution = input("Enter resolution value (default: 100): ").strip()
                resolution = int(resolution) if resolution.isdigit() else 100
                
                filename = input("Enter base filename (default: model): ").strip() or "model"
                
                print("\nExporting model to selected formats...")
                
                # Export to each selected format
                exported_files = {}
                for fmt in formats:
                    output_path = exporter.export_model(
                        scad_code=scad_code,
                        filename=filename,
                        export_format=fmt,
                        resolution=resolution
                    )
                    
                    # Create a proper ExportResult
                    export_result = ExportResult(
                        success=output_path is not None,
                        format=fmt,
                        file_path=output_path,
                        file_size=os.path.getsize(output_path) if output_path and os.path.exists(output_path) else None
                    )
                    
                    # Add to exported_files dictionary
                    exported_files[fmt] = export_result
                
                # Print export results
                print("\nExport Results:")
                for fmt, result_info in exported_files.items():
                    if result_info.success:
                        print(f"  - {fmt}: Success! Saved to {result_info.file_path}")
                    else:
                        print(f"  - {fmt}: Failed")
                
                # Include export results in the returned data
                result["exported_files"] = exported_files
                    
            # The rest of the return statement remains the same
            return {
                "generated_code": result,
                "debug_info": {
                    **debug_info,
                    "stage": "code_generation",
                    "success": True,
                    "code_length": len(scad_code),
                    "add_to_kb": add_to_kb == 'y',
                    "parameters_tuned": tune_params == 'y',
                    "exported_model": export_model == 'y',
                    "export_formats": formats if export_model == 'y' else None,
                    "validation": {
                        "is_valid_syntax": is_valid_syntax,
                        "is_valid_code": valid_code,
                        "error_count": sum(1 for msg in validation_messages if "Error:" in msg),
                        "warning_count": sum(1 for msg in validation_messages if "Warning:" in msg)
                    }
                }
            }
        except Exception as e:
            error_msg = f"Error in generate_scad_code: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "generated_code": {
                    "success": False,
                    "error": error_msg
                },
                "debug_info": {
                    **debug_info,
                    "stage": "code_generation",
                    "success": False,
                    "error": error_msg
                }
            }

    # Return the inner function
    return generate_scad_code

# Helper functions for model preview generation

def get_preview_settings():
    """Get user input for preview settings.
    
    Returns:
        dict: Preview settings including resolution, camera parameters, and image dimensions
    """
    print("\nPreview Settings:")
    print("-" * 30)
    
    # Get resolution
    resolution_input = input("Enter resolution value ($fn) [default: 100]: ").strip()
    resolution = int(resolution_input) if resolution_input.isdigit() else 100
    
    # Ask if user wants to use custom camera settings
    custom_camera = input("Do you want to use custom camera settings? (y/n) [default: n]: ").lower().strip() == 'y'
    
    if custom_camera:
        print("\nCamera settings are in format: translateX,translateY,translateZ,rotateX,rotateY,rotateZ,distance")
        print("Default is: 0,0,0,55,0,25,140")
        
        camera_input = input("Enter camera parameters: ").strip()
        # Validate camera input format (should be 7 comma-separated numbers)
        if camera_input and len(camera_input.split(',')) == 7 and all(part.replace('-', '', 1).replace('.', '', 1).isdigit() for part in camera_input.split(',')):
            camera_params = camera_input
        else:
            print("Invalid camera parameters. Using default settings.")
            camera_params = "0,0,0,55,0,25,140"
    else:
        camera_params = "0,0,0,55,0,25,140"
    
    # Get image dimensions
    custom_dimensions = input("Do you want to use custom image dimensions? (y/n) [default: n]: ").lower().strip() == 'y'
    
    if custom_dimensions:
        width_input = input("Enter image width (pixels) [default: 800]: ").strip()
        width = int(width_input) if width_input.isdigit() and int(width_input) > 0 else 800
        
        height_input = input("Enter image height (pixels) [default: 600]: ").strip()
        height = int(height_input) if height_input.isdigit() and int(height_input) > 0 else 600
    else:
        width = 800
        height = 600
    
    return {
        "resolution": resolution,
        "camera_params": camera_params,
        "width": width,
        "height": height
    }

def generate_model_preview(scad_code: str, filename: str = "preview", 
                     resolution: int = 100, camera_params: str = "0,0,0,55,0,25,140", 
                     width: int = 800, height: int = 600) -> Optional[str]:
    """Generate a preview image of the model
    
    Args:
        scad_code (str): The OpenSCAD code to render
        filename (str): Base name for the output file (without extension)
        resolution (int): Resolution for curves ($fn value)
        camera_params (str): Camera position for the render
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        Optional[str]: Path to the generated preview image or None if failed
    """
    try:
        # Import ModelExporter
        from model_exporter import ModelExporter
        
        # Initialize model exporter with default output directory
        model_exporter = ModelExporter(output_dir="output")
        
        print(f"Generating preview with resolution {resolution} and dimensions {width}x{height}...")
        
        # Use the model exporter to generate the preview
        preview_path = model_exporter.generate_preview(
            scad_code=scad_code,
            filename=filename,
            resolution=resolution,
            camera_params=camera_params,
            width=width,
            height=height
        )
        
        return preview_path
    except Exception as e:
        error_msg = f"Error generating model preview: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        return None

# Helper functions for code extraction

def attempt_to_fix_raw_response(content):
    """
    Attempt to fix the raw response by adding or correcting code tags.
    This is a fallback for when the LLM doesn't properly format its response.
    
    Args:
        content: The raw response from the LLM
        
    Returns:
        Fixed response with proper code tags if fixed, or original content if not
    """
    # Check if there's something that looks like OpenSCAD code but no tags
    openscad_patterns = ['module ', 'function ', 'cube(', 'sphere(', 'cylinder(', 'translate(', 'rotate(', 
                         'scale(', 'difference(', 'union(', 'intersection(', 'color(', 'linear_extrude(', 
                         '$fn=', 'hull(', 'minkowski(']
    
    # If we have no code tags but have OpenSCAD patterns, try to add tags
    if '```' not in content and '<code>' not in content:
        pattern_matches = 0
        for pattern in openscad_patterns:
            if pattern in content:
                pattern_matches += 1
        
        # If we have enough OpenSCAD patterns, add tags around the content
        if pattern_matches >= 3:
            print(f"Found {pattern_matches} OpenSCAD patterns without tags, adding tags")
            return f"```scad\n{content}\n```"
    
    # Check if we have incomplete tags and try to fix them
    if '```scad' in content and '```' not in content[content.find('```scad') + 7:]:
        print("Found incomplete code block, adding closing tag")
        return content + "\n```"
        
    if '<code>' in content and '</code>' not in content:
        print("Found incomplete code tag, adding closing tag")
        return content + "\n</code>"
        
    # Check if content contains code without proper formatting
    code_indicators = ['Here is the code', 'Here\'s the code', 'The OpenSCAD code', 'OpenSCAD implementation']
    for indicator in code_indicators:
        if indicator in content:
            indicator_pos = content.find(indicator)
            next_paragraph_start = content.find('\n\n', indicator_pos)
            
            if next_paragraph_start > indicator_pos:
                # Look for the actual code after the indicator
                potential_code = content[next_paragraph_start:].strip()
                
                # Count OpenSCAD patterns in the potential code
                pattern_matches = 0
                for pattern in openscad_patterns:
                    if pattern in potential_code:
                        pattern_matches += 1
                
                # If there are enough patterns, this is likely code
                if pattern_matches >= 3:
                    print(f"Found {pattern_matches} OpenSCAD patterns after '{indicator}', wrapping with code tags")
                    fixed_content = content[:next_paragraph_start] + "\n\n```scad\n" + potential_code + "\n```"
                    return fixed_content
    
    # Return original content if we couldn't fix it
    return content

def looks_like_complete_openscad(content):
    """
    Check if the content looks like a complete OpenSCAD program even without tags.
    
    Args:
        content: The text to check
        
    Returns:
        Boolean indicating whether the content appears to be a complete OpenSCAD program
    """
    # Core OpenSCAD elements that indicate a complete program
    core_patterns = [
        # Basic shapes
        'cube(', 'sphere(', 'cylinder(', 'polyhedron(',
        # 2D shapes
        'circle(', 'square(', 'polygon(',
        # Operations
        'union(', 'difference(', 'intersection(', 
        # Transformations
        'translate(', 'rotate(', 'scale(', 'mirror(', 'multmatrix(', 'color('
    ]
    
    # Program structure elements
    structure_patterns = [
        'module ', 'function ', '};', '){', '() {', 
        '$fn=', 'import(', 'include <', 'use <'
    ]
    
    # Variables and parameters
    variable_patterns = [
        'let(', 'for(', 'if(', 'else {', '= [', '= "', 
        'true;', 'false;', 'undef;'
    ]
    
    # Count matches of different types of patterns
    core_matches = sum(1 for pattern in core_patterns if pattern in content)
    structure_matches = sum(1 for pattern in structure_patterns if pattern in content)
    variable_matches = sum(1 for pattern in variable_patterns if pattern in content)
    
    # Check for balanced curly braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    balanced_braces = abs(open_braces - close_braces) <= 1  # Allow for one missing brace
    
    # Check for multiple semicolons (statement terminators in OpenSCAD)
    semicolons = content.count(';')
    
    # Heuristic: If we have enough core elements, some structure elements, 
    # relatively balanced braces, and multiple semicolons, it's likely OpenSCAD
    is_likely_openscad = (
        core_matches >= 2 and 
        structure_matches >= 1 and
        variable_matches >= 1 and
        balanced_braces and
        semicolons >= 3 and
        len(content.strip()) > 50  # Not too short
    )
    
    if is_likely_openscad:
        print(f"Content appears to be complete OpenSCAD code: {core_matches} core elements, {structure_matches} structure elements, {semicolons} semicolons")
    
    return is_likely_openscad