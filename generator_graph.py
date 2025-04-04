from langgraph.graph import StateGraph, START, END
from graph_state_tools import (
    State, process_input, extract_keywords, create_search_queries, 
    perform_search, grade_web_content, create_step_back_analyzer,
    hallucination_checker, step_back_analysis_grader,
    step_back_hallucination_router, step_back_quality_router,
    create_examples_retriever, analyze_query, create_scad_code_generator
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
import time


class Model_Generator_Graph:
    def __init__(self, llm_provider, knowledge_base):
        """Initialize the 3D Model Generator with search tools and graph setup"""
        self.memory = MemorySaver()
        self.llm_provider = llm_provider
        self.knowledge_base = knowledge_base
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the processing graph for 3D model generation"""
        workflow = StateGraph(State)
        
        # Create the specialized node functions with our LLM
        step_back_analyzer = create_step_back_analyzer(self.llm_provider)
        examples_retriever = create_examples_retriever(self.knowledge_base)
        scad_code_generator = create_scad_code_generator(self.llm_provider, self.knowledge_base)
        
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
        # workflow.add_node("check_hallucinations", hallucination_checker)  # Commenting out hallucination checker
        workflow.add_node("grade_step_back_analysis", step_back_analysis_grader)
        
        # Knowledge base retrieval
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("retrieve_examples", examples_retriever)
        
        # SCAD code generation (new node)
        workflow.add_node("generate_scad_code", scad_code_generator)
        
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
        # Skip hallucination check and go directly to grading
        workflow.add_edge("run_step_back_analysis", "grade_step_back_analysis")
        
        # Comment out the hallucination check conditional routing
        # workflow.add_edge("run_step_back_analysis", "check_hallucinations")
        # workflow.add_conditional_edges(
        #     "check_hallucinations",
        #     step_back_hallucination_router,
        #     {
        #         "grade_step_back_analysis": "grade_step_back_analysis",
        #         "run_step_back_analysis": "run_step_back_analysis"  # Loop back to step back analysis
        #     }
        # )
        
        # Add conditional routing based on quality check
        workflow.add_conditional_edges(
            "grade_step_back_analysis",
            step_back_quality_router,
            {
                "analyze_query": "analyze_query",
                "run_step_back_analysis": "run_step_back_analysis"  # Loop back to step back analysis
            }
        )
        
        # Knowledge base retrieval
        workflow.add_edge("analyze_query", "retrieve_examples")
        
        # Connect to SCAD code generation (modified from original)
        workflow.add_edge("retrieve_examples", "generate_scad_code")
        workflow.add_edge("generate_scad_code", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def generate(self, input_text):
        """Generate 3D model information based on input text"""
        config = {
            "thread_id": str(uuid.uuid4()),  # Unique identifier for this run
            "checkpoint_ns": "3d_model_gen",  # Namespace for checkpoints
            "checkpoint_id": str(int(time.time()))  # Unique checkpoint ID
        }
        
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=input_text)],
            },
            config=config
        )
        return result