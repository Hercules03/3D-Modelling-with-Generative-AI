from langgraph.graph import StateGraph, START, END
from graph_state_tools import (
    State, process_input, extract_keywords, create_search_queries, 
    perform_search, grade_web_content, create_step_back_analyzer,
    step_back_analysis_grader, step_back_quality_router,
    analyze_query
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
import time
import os
from scad_code_validator import validate_scad_code

class Manual_Knowledge_Graph:
    def __init__(self, llm_provider, knowledge_base):
        """Initialize the Manual Knowledge Input Graph with necessary tools"""
        self.memory = MemorySaver()
        self.llm_provider = llm_provider
        self.knowledge_base = knowledge_base
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the processing graph for manual knowledge input"""
        workflow = StateGraph(State)
        
        # Create the specialized node functions with our LLM
        step_back_analyzer = create_step_back_analyzer(self.llm_provider)
        
        # Add nodes to graph
        # Initial query processing
        workflow.add_node("process_input", process_input)
        workflow.add_node("extract_keywords", extract_keywords)
        workflow.add_node("create_search_queries", create_search_queries)
        
        # Web search for information
        workflow.add_node("perform_search", perform_search)
        workflow.add_node("grade_web_content", grade_web_content)
        
        # Step-back analysis with quality control
        workflow.add_node("run_step_back_analysis", step_back_analyzer)
        workflow.add_node("grade_step_back_analysis", step_back_analysis_grader)
        
        # Knowledge base analysis and storage
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("store_knowledge", self._store_knowledge)
        
        # Add edges between nodes
        # Keyword Extraction
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "extract_keywords")
        workflow.add_edge("extract_keywords", "create_search_queries")
        
        # Web Search of object background information
        workflow.add_edge("create_search_queries", "perform_search")
        workflow.add_edge("perform_search", "grade_web_content")
        
        # Step-back analysis with quality control loops
        workflow.add_edge("grade_web_content", "run_step_back_analysis")
        workflow.add_edge("run_step_back_analysis", "grade_step_back_analysis")
        
        # Add conditional routing based on quality check
        workflow.add_conditional_edges(
            "grade_step_back_analysis",
            step_back_quality_router,
            {
                "analyze_query": "analyze_query",
                "run_step_back_analysis": "run_step_back_analysis"  # Loop back to step back analysis
            }
        )
        
        # Knowledge base storage
        workflow.add_edge("analyze_query", "store_knowledge")
        workflow.add_edge("store_knowledge", END)
        
        # Debug function to print the final state
        def debug_end_state(state):
            print("\nDEBUG - Final state at END node:")
            print(f"State type: {type(state)}")
            print(f"State content: {state}")
            # Only print a few key fields to avoid overwhelming
            success = state.get("success", "NOT_FOUND")
            print(f"Success value: {success}")
            return state
            
        # Add a node to debug the final state
        #workflow.add_node("debug_end", debug_end_state)
        workflow.add_edge("store_knowledge", END)
        #workflow.add_edge("debug_end", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _store_knowledge(self, state: State) -> dict:
        """Store the analyzed knowledge along with user-provided SCAD code"""
        input_text = state.get("input_text", "")
        step_back_analysis = state.get("step_back_analysis", {})
        query_analysis = state.get("query_analysis", {})
        extracted_keywords = state.get("extracted_keywords", {})
        debug_info = state.get("debug_info", {})
        
        print("\n" + "="*50)
        print("SAVING MANUAL KNOWLEDGE")
        print("="*50)
        
        try:
            # Read SCAD code from add.scad file
            scad_file_path = "add.scad"
            if not os.path.exists(scad_file_path):
                error_msg = f"Error: {scad_file_path} file not found. Please create this file with your OpenSCAD code."
                print(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "debug_info": {
                        **debug_info,
                        "stage": "store_knowledge",
                        "success": False,
                        "error": error_msg
                    }
                }
            
            with open(scad_file_path, "r") as f:
                scad_code = f.read().strip()
            
            if not scad_code:
                error_msg = f"Error: {scad_file_path} is empty. Please add your OpenSCAD code to the file."
                print(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "debug_info": {
                        **debug_info,
                        "stage": "store_knowledge",
                        "success": False,
                        "error": error_msg
                    }
                }
            
            print(f"\nRead OpenSCAD code from {scad_file_path}:")
            print("-" * 50)
            print(f"{scad_code[:500]}..." if len(scad_code) > 500 else scad_code)
            print("-" * 50)
            
            # Validate the SCAD code
            is_valid, validation_messages = validate_scad_code(scad_code)
            
            print("\nValidation Results:")
            for message in validation_messages:
                print(f"- {message}")
                
            # If code is not valid, ask for confirmation to proceed
            if not is_valid:
                print("\nWarning: The OpenSCAD code has validation errors.")
                confirmation = input("Do you want to proceed anyway? (y/n): ").lower().strip()
                if confirmation != 'y':
                    error_msg = "Manual knowledge input cancelled due to code validation issues."
                    print(f"\n{error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "validation_messages": validation_messages,
                        "debug_info": {
                            **debug_info,
                            "stage": "store_knowledge",
                            "success": False,
                            "error": error_msg,
                            "validation_failed": True
                        }
                    }
            
            
            # Prepare metadata for knowledge base
            core_type = extracted_keywords.get('core_type', '')
            compound_type = extracted_keywords.get('compound_type', '')
            modifiers = extracted_keywords.get('modifiers', [])
            
            # Get techniques and other attributes from query analysis
            techniques_needed = query_analysis.get('techniques_needed', [])
            style_preference = query_analysis.get('style_preference', 'Parametric')
            complexity = query_analysis.get('complexity', 'MEDIUM')
            
            # Prepare the metadata
            metadata = {
                "object_type": compound_type if compound_type else core_type,
                "features": modifiers,
                "step_back_analysis": step_back_analysis,
                "geometric_properties": [],  # Can be extracted later
                "techniques_used": techniques_needed,
                "style": style_preference,
                "complexity": complexity,
                "query_analysis": query_analysis
            }
            
            # Add to knowledge base
            print("\nAdding example to knowledge base...")
            result = self.knowledge_base.add_example(input_text, scad_code, metadata)
            
            if result:
                print("\nManual knowledge successfully added to the knowledge base!")
                return_value = {
                    "success": True,
                    "debug_info": {
                        **debug_info,
                        "stage": "store_knowledge",
                        "success": True
                    }
                }
                print("\nDEBUG - Returning from _store_knowledge (success):")
                print(f"Return value: {return_value}")
                return return_value
            else:
                error_msg = "Failed to add manual knowledge to the knowledge base."
                print(f"\n{error_msg}")
                return_value = {
                    "success": False,
                    "error": error_msg,
                    "debug_info": {
                        **debug_info,
                        "stage": "store_knowledge",
                        "success": False,
                        "error": error_msg
                    }
                }
                print("\nDEBUG - Returning from _store_knowledge (failure):")
                print(f"Return value: {return_value}")
                return return_value
            
        except Exception as e:
            error_msg = f"Error storing knowledge: {str(e)}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            return_value = {
                "success": False,
                "error": error_msg,
                "debug_info": {
                    **debug_info,
                    "stage": "store_knowledge",
                    "success": False,
                    "error": error_msg
                }
            }
            print("\nDEBUG - Returning from _store_knowledge (exception):")
            print(f"Return value: {return_value}")
            return return_value
    
    def process_manual_knowledge(self, input_text):
        """Process manual knowledge input based on the description"""
        config = {
            "thread_id": str(uuid.uuid4()),  # Unique identifier for this run
            "checkpoint_ns": "manual_knowledge",  # Namespace for checkpoints
            "checkpoint_id": str(int(time.time()))  # Unique checkpoint ID
        }
        
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=input_text)],
            },
            config=config
        )
        print("\nDEBUG - Result from graph.invoke:")
        print(f"Result type: {type(result)}")
        print(f"Result content: {result}")
        return result
        
    def handle_manual_knowledge_input(self):
        """Handle the complete process of manual knowledge input in a user-friendly way"""
        print("\n" + "="*50)
        print("MANUAL KNOWLEDGE INPUT")
        print("="*50)
        
        # Step 1: Check for the add.scad file
        scad_file_path = "add.scad"
        if not os.path.exists(scad_file_path):
            print("\nCreating empty add.scad file. Please add your OpenSCAD code to this file.")
            with open(scad_file_path, "w") as f:
                f.write("// Add your OpenSCAD code here\n")
        else:
            print(f"\nFound existing add.scad file at: {os.path.abspath(scad_file_path)}")
            
            # Check if file has content
            try:
                with open(scad_file_path, "r") as f:
                    code = f.read().strip()
                if not code or code == "// Add your OpenSCAD code here":
                    print("\nThe add.scad file appears to be empty or contains only the template comment.")
                    print("Please add your OpenSCAD code to this file before proceeding.")
                else:
                    print(f"The file contains {len(code.splitlines())} lines of code.")
            except Exception as e:
                print(f"\nWarning: Could not read the add.scad file: {str(e)}")
        
        # Step 2: Ask user to confirm that they've added their code
        while True:
            ready = input("\nHave you added your OpenSCAD code to the add.scad file? (y/n): ").lower().strip()
            if ready == 'y':
                break
            elif ready == 'n':
                print("\nPlease add your OpenSCAD code to the add.scad file before continuing.")
                editor_cmd = input("\nWould you like help opening the file in a text editor? (y/n): ").lower().strip()
                if editor_cmd == 'y':
                    print(f"\nFile location: {os.path.abspath(scad_file_path)}")
                    print("You can open this file in your preferred text editor.")
                continue_anyway = input("\nDo you want to continue anyway? (y/n): ").lower().strip()
                if continue_anyway != 'y':
                    print("\nManual knowledge input cancelled.")
                    return {"success": False, "error": "Cancelled by user"}
                break
        
        # Step 3: Get description from user
        while True:
            description = input("\nEnter a detailed description of the 3D object in the add.scad file: ").strip()
            if description:
                break
            print("Description cannot be empty. Please provide a description.")
        
        # Step 4: Run the processing graph
        print("\nProcessing your description. This may take a moment...")
        result = self.process_manual_knowledge(description)
        
        print("\nDEBUG - Result in handle_manual_knowledge_input:")
        print(f"Result type: {type(result)}")
        print(f"Result content: {result}")
        print(f"Success value: {result.get('success', 'NOT_FOUND')}")
        
        # Step 5: Report result to user
        # The success status is nested inside debug_info
        debug_info = result.get("debug_info", {})
        success = debug_info.get("success", False)
        
        if success:
            print("\nSuccess! Your knowledge has been added to the database.")
            print("\nYou can view it in the knowledge base explorer (option 4 from the main menu).")
        else:
            error = debug_info.get("error", "Unknown error")
            print(f"\nFailed to add knowledge: {error}")
            
            # If validation failed, show more details
            if "validation_messages" in result:
                print("\nCode validation messages:")
                for message in result["validation_messages"]:
                    print(f"- {message}")
        
        return result