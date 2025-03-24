import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from constant import *
from KeywordExtractor import KeywordExtractor
from conversation_logger import ConversationLogger
from datetime import datetime
from prompts import KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT, STEP_BACK_PROMPT_TEMPLATE
from LLM import LLMProvider
from metadata_extractor import MetadataExtractor
import re
import hashlib

class EnhancedSCADKnowledgeBase:
    # Add this as a class variable at the top of the class
    STANDARD_CATEGORIES = {
        "container": {
            "description": "Objects that can hold or contain other items",
            "examples": ["cup", "mug", "bowl", "vase", "bottle", "box", "jar", "tray", "basket", "planter", "container"]
        },
        "furniture": {
            "description": "Objects used for living or working spaces",
            "examples": ["table", "chair", "desk", "shelf", "cabinet", "bench", "stand", "organizer", "rack", "holder"]
        },
        "decorative": {
            "description": "Objects primarily for aesthetic purposes",
            "examples": ["sculpture", "ornament", "statue", "figurine", "vase", "frame", "display", "art", "decoration"]
        },
        "functional": {
            "description": "Utility objects with specific practical purposes",
            "examples": ["holder", "stand", "bracket", "hook", "hanger", "clip", "mount", "support", "organizer", "adapter"]
        },
        "geometric": {
            "description": "Basic or complex geometric shapes and mathematical objects",
            "examples": ["cube", "sphere", "cylinder", "cone", "polyhedron", "torus", "prism", "pyramid", "helix"]
        },
        "mechanical": {
            "description": "Parts or components of mechanical systems",
            "examples": ["gear", "bolt", "nut", "bearing", "lever", "hinge", "joint", "wheel", "pulley", "spring"]
        },
        "enclosure": {
            "description": "Objects designed to enclose or protect other items",
            "examples": ["case", "box", "housing", "cover", "shell", "enclosure", "protection", "sleeve", "cap"]
        },
        "modular": {
            "description": "Objects designed to be combined or connected with others",
            "examples": ["connector", "adapter", "joint", "mount", "tile", "block", "module", "segment", "piece"]
        },
        "other": {
            "description": "Objects that don't fit in other categories",
            "examples": []
        }
    }

    STANDARD_PROPERTIES = {
        # Physical dimensions and characteristics
        "size": ["tiny", "small", "medium", "large", "huge"],
        "wall_thickness": ["thin", "medium", "thick", "solid"],
        "hollow": ["yes", "no", "partial"],
        
        # Design characteristics
        "style": ["modern", "traditional", "minimalist", "decorative", "industrial", "organic", "geometric", "artistic"],
        "complexity": ["simple", "moderate", "complex", "intricate"],
        "symmetry": ["radial", "bilateral", "asymmetric", "periodic"],
        
        # 3D Printing specific
        "printability": ["easy", "moderate", "challenging", "requires_support"],
        "orientation": ["flat", "vertical", "angled", "any"],
        "support_needed": ["none", "minimal", "moderate", "extensive"],
        "infill_requirement": ["low", "medium", "high", "solid"],
        
        # Assembly and structure
        "assembly": ["single_piece", "multi_part", "snap_fit", "threaded", "interlocking"],
        "stability": ["self_standing", "needs_support", "wall_mounted", "hanging"],
        "adjustability": ["fixed", "adjustable", "modular", "customizable"],
        
        # Material and finishing
        "material_compatibility": ["any", "pla", "abs", "petg", "resin", "multi_material"],
        "surface_finish": ["smooth", "textured", "patterned", "functional"],
        "post_processing": ["none", "minimal", "required", "optional"],
        
        # Functional aspects
        "durability": ["light_duty", "medium_duty", "heavy_duty"],
        "water_resistance": ["none", "splash_proof", "water_tight"],
        "heat_resistance": ["low", "medium", "high"],
        
        # Design intent
        "customization": ["fixed", "parametric", "highly_customizable"],
        "purpose": ["practical", "decorative", "educational", "prototype"]
    }

    def __init__(self):
        """Initialize the enhanced knowledge base with ChromaDB"""
        print("\n" + "="*50)
        print("Initializing Enhanced SCAD Knowledge Base")
        print("="*50)
        
        # Set up paths and basic components
        print("\nSetting up basic components...")
        self.persistence_dir = os.path.join(os.getcwd(), "scad_knowledge_base/chroma")
        self.logger = ConversationLogger()
        print(f"- Persistence directory: {self.persistence_dir}")
        print("- Conversation logger initialized")
        
        # Initialize LLM and metadata extractor
        print("\nInitializing LLM provider...")
        self.llm = LLMProvider.get_llm()
        self.metadata_extractor = MetadataExtractor()
        print("- LLM provider initialized")
        print("- Metadata extractor initialized")
        
        # Set up prompts
        print("\nInitializing prompts...")
        self.keyword_prompt = KEYWORD_EXTRACTOR_PROMPT
        print("- Keyword extraction prompt loaded")
        
        # Initialize ChromaDB
        print("\nInitializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=self.persistence_dir)
            print("- ChromaDB client initialized")
            
            # Initialize embedding function
            self.embedding_function = SentenceTransformerEmbeddingFunction()
            print("- Sentence transformer embedding function initialized")
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name="scad_examples",
                    embedding_function=self.embedding_function
                )
                example_count = self.collection.count()
                print(f"- Using existing collection 'scad_examples' with {example_count} examples")
                
            except ValueError:
                print("- Creating new collection 'scad_examples'...")
                self.collection = self.client.create_collection(
                    name="scad_examples",
                    embedding_function=self.embedding_function
                )
                print("- New collection created successfully")
                example_count = 0
            
            # Load timestamp
            self.last_processed_time = self._load_last_processed_time()
            print(f"- Last processed timestamp: {self.last_processed_time}")
            
            # Load new examples
            print("\nChecking for new examples...")
            new_count = self._load_new_examples()
            if new_count > 0:
                print(f"- Added {new_count} new examples")
                print(f"- Collection now has {self.collection.count()} total examples")
            else:
                print("- No new examples found")
            
            print("\n" + "="*50)
            print(f"Knowledge Base Ready | {example_count} examples loaded")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\nError initializing ChromaDB: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            print("\n=== Cleaning up Enhanced SCAD Knowledge Base ===")
            
            # Save the last processed timestamp
            print("- Saving last processed timestamp...")
            self._save_last_processed_time()
            
            # Close the client (PersistentClient handles persistence automatically)
            print("- Closing ChromaDB client...")
            if hasattr(self, 'client'):
                # Just set to None instead of trying to reset
                self.collection = None
                self.client = None
            
            print("=== Cleanup Complete ===\n")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def _load_last_processed_time(self):
        """Load the timestamp of last processed example"""
        try:
            with open(f"{self.persistence_dir}/last_processed.json", "r") as f:
                return json.load(f)["timestamp"]
        except (FileNotFoundError, json.JSONDecodeError):
            return "1970-01-01T00:00:00"  # Start from epoch if no timestamp
    
    def _save_last_processed_time(self):
        """Save the timestamp of last processed example"""
        os.makedirs(self.persistence_dir, exist_ok=True)
        with open(f"{self.persistence_dir}/last_processed.json", "w") as f:
            json.dump({"timestamp": self.last_processed_time}, f)
    
    def _load_new_examples(self):
        """Load only new examples from conversation logs"""
        try:
            # Get all examples from logs
            examples = self.logger.get_scad_generation_logs()
            if not examples:
                return 0
            
            new_examples = 0
            for example in examples:
                # Skip if example doesn't have a timestamp or wasn't accepted by user
                example_time = example.get('timestamp', '1970-01-01T00:00:00')
                if example_time <= self.last_processed_time:
                    continue
                    
                # Skip if not a SCAD generation example or not accepted by user
                if not example.get('user_accepted', False) or example.get('type') != 'scad_generation':
                    continue
                
                description = example.get('request', '')
                code = example.get('code', '')
                
                if not description or not code:
                    continue
                
                # Generate unique ID based on content hash
                content_hash = hashlib.md5(f"{description}{code}".encode()).hexdigest()[:8]
                example_id = f"{self._generate_base_name(description)}_{content_hash}"
                
                # Check if example already exists
                try:
                    self.collection.get(ids=[example_id])
                    continue
                except Exception:
                    # Example doesn't exist, add it
                    try:
                        self.collection.add(
                            documents=[description],
                            metadatas=[{
                                "code": code,
                                "timestamp": example_time,
                                "type": "scad_generation",
                                "user_accepted": True
                            }],
                            ids=[example_id]
                        )
                        new_examples += 1
                        
                        # Update last processed timestamp if this example is newer
                        if example_time > self.last_processed_time:
                            self.last_processed_time = example_time
                            self._save_last_processed_time()
                            
                    except Exception:
                        continue
            
            return new_examples
                
        except Exception as e:
            print(f"Error loading new examples: {str(e)}")
            return 0
    
    def _generate_base_name(self, description):
        """Generate a base name for the example"""
        # Convert to lowercase and split into words
        words = description.lower().split()
        
        # Common words to ignore
        stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she'
        }
        
        # Find first meaningful word
        for word in words:
            if word not in stop_words:
                return ''.join(c for c in word if c.isalnum())
        
        return 'example'
    
    def _analyze_object_categories(self, object_type, description):
        """Use LLM to analyze object categories using standardized categories and properties"""
        try:
            # Convert standard categories to a format suitable for the prompt
            categories_info = "\n".join(
                f"- {cat}: {info['description']} (e.g., {', '.join(info['examples'])})"
                for cat, info in self.STANDARD_CATEGORIES.items()
            )
            
            properties_info = "\n".join(
                f"- {prop}: {', '.join(values)}"
                for prop, values in self.STANDARD_PROPERTIES.items()
            )

            # Use the prompt template from prompts.py
            prompt = CATEGORY_ANALYSIS_PROMPT.format(
                object_type=object_type,
                description=description,
                categories_info=categories_info,
                properties_info=properties_info
            )

            response = self.llm.invoke(prompt)
            
            # Ensure we're getting a clean JSON string
            json_str = response.content.strip()
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1]
            if json_str.endswith("```"):
                json_str = json_str.rsplit("```", 1)[0]
            json_str = json_str.strip()
            
            try:
                result = json.loads(json_str)
                
                # Validate and clean the response
                cleaned_result = {
                    "categories": [
                        cat for cat in result.get("categories", [])
                        if cat in self.STANDARD_CATEGORIES
                    ],
                    "properties": {
                        prop: value
                        for prop, value in result.get("properties", {}).items()
                        if prop in self.STANDARD_PROPERTIES
                        and value in self.STANDARD_PROPERTIES[prop]
                    },
                    "similar_objects": [
                        obj for obj in result.get("similar_objects", [])
                        if any(obj in cat_info["examples"]
                              for cat_info in self.STANDARD_CATEGORIES.values())
                    ]
                }
                
                # Ensure at least one category is assigned
                if not cleaned_result["categories"]:
                    cleaned_result["categories"] = ["container"]  # Default to container for most 3D objects
                
                return cleaned_result
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {str(e)}")
                print(f"Raw response: {json_str}")
                return {
                    "categories": ["container"],
                    "properties": {},
                    "similar_objects": []
                }
                
        except Exception as e:
            print(f"Error analyzing object categories: {e}")
            return {
                "categories": ["container"],
                "properties": {},
                "similar_objects": []
            }

    def add_example(self, description, code, metadata=None):
        """Add a new example to the knowledge base"""
        try:
            # Generate a unique ID based on style and hash of description
            style = metadata.get('style', 'unknown').lower() if metadata else 'unknown'
            description_hash = hashlib.md5(description.encode()).hexdigest()[:8]
            example_id = f"{style}_{description_hash}"
            
            # Check if example already exists
            if self._example_exists(example_id):
                print(f"\nSkipping duplicate example with ID: {example_id}")
                print("This example is already in the knowledge base.")
                return True
            
            # Extract metadata if not provided
            if not metadata:
                metadata = self.metadata_extractor.extract_metadata(description)
            
            # Perform step-back analysis
            step_back_prompt = STEP_BACK_PROMPT_TEMPLATE.format(query=description)
            step_back_response = self.llm.invoke(step_back_prompt)
            technical_analysis = step_back_response.content
            
            # Parse step-back analysis
            principles = []
            abstractions = []
            approach = []
            
            current_section = None
            for line in technical_analysis.split('\n'):
                line = line.strip()
                if 'CORE PRINCIPLES:' in line:
                    current_section = 'principles'
                elif 'SHAPE COMPONENTS:' in line:
                    current_section = 'abstractions'
                elif 'IMPLEMENTATION STEPS:' in line:
                    current_section = 'approach'
                elif line and line[0] == '-' and current_section == 'principles':
                    principles.append(line[1:].strip())
                elif line and line[0] == '-' and current_section == 'abstractions':
                    abstractions.append(line[1:].strip())
                elif line and line[0].isdigit() and current_section == 'approach':
                    approach.append(line[line.find('.')+1:].strip())
            
            # Add step-back analysis to metadata
            metadata['step_back_analysis'] = {
                'principles': principles,
                'abstractions': abstractions,
                'approach': approach
            }
            
            # Flatten metadata for ChromaDB
            flattened_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # For dictionaries, create separate keys with prefixes
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (list, dict)):
                            flattened_metadata[f"{key}_{sub_key}"] = json.dumps(sub_value)
                        else:
                            flattened_metadata[f"{key}_{sub_key}"] = str(sub_value)
                elif isinstance(value, list):
                    # For lists, join into a comma-separated string
                    flattened_metadata[key] = ", ".join(str(item) for item in value)
                else:
                    flattened_metadata[key] = str(value)
            
            # Add code to metadata
            flattened_metadata['code'] = code
            
            # Add example to ChromaDB
            print(f"\nAdd of existing embedding ID: {example_id}")
            self.collection.add(
                documents=[description],
                metadatas=[flattened_metadata],
                ids=[example_id]
            )
            
            # Update last processed timestamp
            self.last_processed = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error adding example: {str(e)}")
            return False
    
    def get_relevant_examples(self, query, max_examples=2, filters=None, similarity_threshold=0.15):
        """Get relevant examples based on description similarity and metadata filters"""
        try:
            print(f"\nSearching for examples matching: '{query}'")
            
            # First, perform step-back analysis to enrich the search query
            step_back_prompt = STEP_BACK_PROMPT_TEMPLATE.format(query=query)
            step_back_response = self.llm.invoke(step_back_prompt)
            
            # Extract technical aspects from step-back analysis
            technical_analysis = step_back_response.content
            
            # Parse step-back analysis into structured components
            principles = []
            abstractions = []
            approach = []
            
            current_section = None
            for line in technical_analysis.split('\n'):
                line = line.strip()
                if 'CORE PRINCIPLES:' in line:
                    current_section = 'principles'
                elif 'SHAPE COMPONENTS:' in line:
                    current_section = 'abstractions'
                elif 'IMPLEMENTATION STEPS:' in line:
                    current_section = 'approach'
                elif line and line[0] == '-' and current_section == 'principles':
                    principles.append(line[1:].strip())
                elif line and line[0] == '-' and current_section == 'abstractions':
                    abstractions.append(line[1:].strip())
                elif line and line[0].isdigit() and current_section == 'approach':
                    approach.append(line[line.find('.')+1:].strip())
            
            step_back_components = {
                'principles': principles,
                'abstractions': abstractions,
                'approach': approach
            }
            
            # Create an enriched search query combining original query and technical analysis
            enriched_query = f"""
            Original Request: {query}
            
            Technical Analysis:
            CORE PRINCIPLES:
            {chr(10).join(f"- {p}" for p in principles)}
            
            SHAPE COMPONENTS:
            {chr(10).join(f"- {a}" for a in abstractions)}
            
            IMPLEMENTATION STEPS:
            {chr(10).join(f"{i+1}. {s}" for i, s in enumerate(approach))}
            """
            
            print("\nEnriched search query with technical analysis")
            
            # Extract metadata and analyze components from query
            query_metadata = self.metadata_extractor.extract_metadata(query)
            query_metadata['step_back_analysis'] = step_back_components
            
            # Analyze required components from the query using LLM
            components_prompt = f"""Analyze the following request and identify the key OpenSCAD components needed:
            {query}
            {technical_analysis}
            
            List only the core components needed (modules, primitives, transformations, boolean operations).
            Respond with a valid JSON array of objects, each with 'type' and 'name' fields."""
            
            components_response = self.llm.invoke(components_prompt)
            try:
                # Clean up JSON response
                json_str = components_response.content.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("```", 1)[0]
                query_metadata['components'] = json.loads(json_str.strip())
            except:
                query_metadata['components'] = []
            
            print("\nExtracted metadata and components:")
            for key, value in query_metadata.items():
                if key == 'step_back_analysis':
                    print("\n  📋 Step-Back Analysis:")
                    analysis = value
                    
                    print("\n    🎯 Core Principles:")
                    for principle in analysis['principles']:
                        print(f"      • {principle}")
                    
                    print("\n    🔷 Shape Abstractions:")
                    for abstraction in analysis['abstractions']:
                        print(f"      • {abstraction}")
                    
                    print("\n    🛠️  Implementation Approach:")
                    for i, step in enumerate(analysis['approach'], 1):
                        print(f"      {i}. {step}")
                
                elif key == 'components':
                    print("\n  🧩 Components:")
                    # Group components by type
                    components_by_type = {}
                    for comp in value:
                        comp_type = comp['type']
                        if comp_type not in components_by_type:
                            components_by_type[comp_type] = []
                        components_by_type[comp_type].append(comp['name'])
                    
                    # Print grouped components
                    for comp_type, names in components_by_type.items():
                        print(f"\n    {comp_type.title()}s:")
                        for name in names:
                            print(f"      • {name}")
                else:
                    print(f"  • {key}: {value}")
            
            # First try: Get results without any filtering
            results = self.collection.query(
                query_texts=[enriched_query],
                n_results=10,
                include=['metadatas', 'documents', 'distances']
            )
            
            if not results or not results['documents']:
                print("No examples found in knowledge base.")
                return []
            
            # Rank and filter results
            ranked_results = self._rank_results(results, query_metadata)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in ranked_results[:max_examples]
                if result['scores']['final'] >= similarity_threshold
            ]
            
            if filtered_results:
                print(f"\nFound {len(filtered_results)} relevant examples:")
                for result in filtered_results:
                    print(f"\nSimilarity Score: {result['scores']['final']:.2f}")
                    print(f"Description: {result['description']}")
                    print("\nScore Breakdown:")
                    print(f"  Base Similarity: {result['scores']['similarity']:.3f}")
                    print(f"  Step-back Analysis: {sum(result['scores']['step_back'].values()) / len(result['scores']['step_back']):.3f}")
                    print(f"  Component Match: {result['scores']['component_match']:.3f}")
                    print(f"  Metadata Match: {result['scores']['metadata_match']:.3f}")
                    print(f"  Complexity: {result['scores']['complexity']:.3f}")
            else:
                print(f"\nNo examples met the similarity threshold of {similarity_threshold}")
            
            return filtered_results
            
        except Exception as e:
            print(f"Error getting relevant examples: {str(e)}")
            return []
            
    def _extract_object_type(self, description):
        """Extract the main object type and modifiers from a description.
        
        Args:
            description (str): The input description to analyze
            
        Returns:
            str: The extracted object type (either compound or core type)
        """
        print("\n=== Starting Keyword Extraction ===")
        print(f"Input description: {description}")
        
        try:
            # Create timestamp for the query
            timestamp = datetime.now().isoformat()
            print(f"Created timestamp: {timestamp}")
            
            # Get the response from LLM
            print("Calling LLM with keyword prompt...")
            response = self.llm.invoke(self.keyword_prompt.format(description=description))
            response_content = response.content.strip()
            print(f"LLM Response: {response_content}")
            
            try:
                # Parse the JSON response
                print("Attempting to parse JSON response...")
                parsed_response = json.loads(response_content)
                print(f"Parsed response: {json.dumps(parsed_response, indent=2)}")
                
                # Create the full analysis object
                analysis = {
                    "query": {
                        "input": description,
                        "timestamp": timestamp
                    },
                    "response": {
                        "core_type": parsed_response.get("core_type", ""),
                        "modifiers": parsed_response.get("modifiers", []),
                        "compound_type": parsed_response.get("compound_type", "")
                    },
                    "metadata": {
                        "success": True,
                        "error": None
                    }
                }
                print(f"Created analysis object: {json.dumps(analysis, indent=2)}")
                
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                # Handle case where response is not valid JSON
                analysis = {
                    "query": {
                        "input": description,
                        "timestamp": timestamp
                    },
                    "response": {
                        "raw_content": response_content
                    },
                    "metadata": {
                        "success": False,
                        "error": "Failed to parse JSON response"
                    }
                }
            
            # Store the analysis for reference
            self.last_object_analysis = analysis
            print("Stored analysis in last_object_analysis")
            
            # Log the query-response pair
            print("Attempting to log query-response pair...")
            self.logger.log_keyword_extraction(analysis)
            print("Successfully logged query-response pair")
            
            # Return the appropriate type (compound if available, otherwise core)
            if analysis["metadata"]["success"]:
                result = (analysis["response"]["compound_type"] 
                       if analysis["response"]["compound_type"] 
                       else analysis["response"]["core_type"])
                print(f"Returning result: {result}")
                return result
            else:
                print(f"Returning raw response: {response_content.strip()}")
                return response_content.strip()
                
        except Exception as e:
            print(f"Error in keyword extraction: {str(e)}")
            error_analysis = {
                "query": {
                    "input": description,
                    "timestamp": datetime.now().isoformat()
                },
                "response": {
                    "raw_content": str(e)
                },
                "metadata": {
                    "success": False,
                    "error": f"Exception during extraction: {str(e)}"
                }
            }
            self.last_object_analysis = error_analysis
            print("Attempting to log error analysis...")
            self.logger.log_keyword_extraction(error_analysis)
            print("Successfully logged error analysis")
            return description.strip()
    
    def _generate_base_name(self, description):
        """Generate a base name for the example"""
        # Convert to lowercase and split into words
        words = description.lower().split()
        
        # Common words to ignore
        stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she'
        }
        
        # Find first meaningful word
        for word in words:
            if word not in stop_words:
                return ''.join(c for c in word if c.isalnum())
        
        return 'example'
    
    def _calculate_complexity_score(self, code, metadata):
        """Calculate complexity score based on code analysis and metadata"""
        score = 0
        
        # Code-based complexity factors
        try:
            # Count number of operations/functions
            operations = len([line for line in code.split('\n') 
                            if any(op in line.lower() for op in ['union', 'difference', 'intersection', 'translate', 'rotate', 'scale'])])
            score += min(operations * 2, 30)  # Max 30 points for operations
            
            # Count nested levels
            max_nesting = 0
            current_nesting = 0
            for line in code.split('\n'):
                if '{' in line:
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                if '}' in line:
                    current_nesting -= 1
            score += min(max_nesting * 5, 20)  # Max 20 points for nesting
            
            # Count variables and modules
            variables = len([line for line in code.split('\n') if '=' in line and not line.strip().startswith('//')])
            modules = len([line for line in code.split('\n') if 'module' in line])
            score += min((variables + modules) * 2, 20)  # Max 20 points for variables/modules
            
            # Analyze geometric complexity
            geometric_ops = len([line for line in code.split('\n') 
                               if any(shape in line.lower() for shape in ['sphere', 'cube', 'cylinder', 'polyhedron'])])
            score += min(geometric_ops * 3, 15)  # Max 15 points for geometric operations
            
        except Exception as e:
            print(f"Error calculating code complexity: {str(e)}")
        
        # Metadata-based complexity factors
        try:
            # Consider declared complexity
            if metadata.get('properties_complexity') == 'intricate':
                score += 15
            elif metadata.get('properties_complexity') == 'complex':
                score += 10
            elif metadata.get('properties_complexity') == 'moderate':
                score += 5
            
            # Consider printability
            if metadata.get('properties_printability') == 'challenging':
                score += 10
            elif metadata.get('properties_printability') == 'requires_support':
                score += 5
            
            # Consider support needed
            if metadata.get('properties_support_needed') == 'extensive':
                score += 10
            elif metadata.get('properties_support_needed') == 'moderate':
                score += 5
            
        except Exception as e:
            print(f"Error calculating metadata complexity: {str(e)}")
        
        # Normalize score to 0-100 range
        normalized_score = min(max(score, 0), 100)
        return normalized_score

    def _analyze_components(self, code):
        """Analyze and extract components from SCAD code"""
        components = []
        try:
            # Extract modules (reusable components)
            module_pattern = r'module\s+(\w+)\s*\([^)]*\)\s*{'
            modules = re.findall(module_pattern, code)
            for module in modules:
                components.append({
                    'type': 'module',
                    'name': module,
                    'reusable': True
                })
            
            # Extract main geometric shapes
            shape_pattern = r'(sphere|cube|cylinder|polyhedron)\s*\('
            shapes = re.findall(shape_pattern, code)
            for shape in shapes:
                components.append({
                    'type': 'primitive',
                    'name': shape,
                    'reusable': False
                })
            
            # Extract transformations
            transform_pattern = r'(translate|rotate|scale|mirror)\s*\('
            transformations = re.findall(transform_pattern, code)
            for transform in transformations:
                components.append({
                    'type': 'transformation',
                    'name': transform,
                    'reusable': False
                })
            
            # Extract boolean operations
            bool_pattern = r'(union|difference|intersection)\s*\('
            booleans = re.findall(bool_pattern, code)
            for boolean in booleans:
                components.append({
                    'type': 'boolean',
                    'name': boolean,
                    'reusable': False
                })
            
        except Exception as e:
            print(f"Error analyzing components: {str(e)}")
        
        return components

    def _rank_results(self, results, query_metadata):
        """Rank and filter search results based on multiple criteria"""
        ranked_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Convert ChromaDB distance to similarity score (0-1)
            base_similarity = 1 - min(distance, 1.0)
            
            # Get step-back analysis scores
            step_back_scores = {'principles': 0.0, 'abstractions': 0.0, 'approach': 0.0}
            
            if 'step_back_analysis' in query_metadata and 'step_back_analysis' in metadata:
                query_analysis = query_metadata['step_back_analysis']
                example_analysis = metadata['step_back_analysis']
                
                # Compare principles
                if query_analysis['principles'] and example_analysis['principles']:
                    matches = sum(1 for p1 in query_analysis['principles']
                                for p2 in example_analysis['principles']
                                if self._fuzzy_match(p1, p2))
                    total = max(len(query_analysis['principles']), len(example_analysis['principles']))
                    step_back_scores['principles'] = matches / total if total > 0 else 0.0
                
                # Compare abstractions
                if query_analysis['abstractions'] and example_analysis['abstractions']:
                    matches = sum(1 for a1 in query_analysis['abstractions']
                                for a2 in example_analysis['abstractions']
                                if self._fuzzy_match(a1, a2))
                    total = max(len(query_analysis['abstractions']), len(example_analysis['abstractions']))
                    step_back_scores['abstractions'] = matches / total if total > 0 else 0.0
                
                # Compare approach steps
                if query_analysis['approach'] and example_analysis['approach']:
                    matches = sum(1 for s1 in query_analysis['approach']
                                for s2 in example_analysis['approach']
                                if self._fuzzy_match(s1, s2))
                    total = max(len(query_analysis['approach']), len(example_analysis['approach']))
                    step_back_scores['approach'] = matches / total if total > 0 else 0.0
            
            # Calculate component match score
            component_match = 0.0
            if 'components' in query_metadata and 'components' in metadata:
                query_components = query_metadata['components']
                example_components = metadata['components']
                
                # Convert string components to dict format if needed
                if isinstance(example_components, str):
                    try:
                        example_components = json.loads(example_components)
                    except:
                        example_components = []
                
                if query_components and example_components:
                    matches = 0
                    total_components = len(query_components)
                    
                    for q_comp in query_components:
                        for e_comp in example_components:
                            if (isinstance(q_comp, dict) and isinstance(e_comp, dict) and
                                q_comp['type'] == e_comp['type'] and
                                self._fuzzy_match(q_comp['name'], e_comp['name'])):
                                matches += 1
                                break
                    
                    component_match = matches / total_components if total_components > 0 else 0.0
            
            # Calculate metadata match score
            metadata_match = 0.0
            metadata_scores = []
            
            key_properties = [
                'object_type',
                'style',
                'use_case',
                'geometric_properties',
                'technical_requirements'
            ]
            
            for key in key_properties:
                if key in query_metadata and key in metadata:
                    query_value = query_metadata[key]
                    example_value = metadata[key]
                    
                    # Handle array values
                    if isinstance(query_value, list) and isinstance(example_value, list):
                        matches = sum(1 for q in query_value
                                    for e in example_value
                                    if self._fuzzy_match(str(q), str(e)))
                        total = max(len(query_value), len(example_value))
                        score = matches / total if total > 0 else 0.0
                    else:
                        # Handle string values
                        score = 1.0 if self._fuzzy_match(str(query_value), str(example_value)) else 0.0
                    
                    metadata_scores.append(score)
            
            metadata_match = sum(metadata_scores) / len(metadata_scores) if metadata_scores else 0.0
            
            # Calculate complexity score
            complexity_score = 0.0
            if 'complexity' in query_metadata and 'complexity' in metadata:
                complexity_map = {'SIMPLE': 0.0, 'MEDIUM': 0.5, 'COMPLEX': 1.0}
                query_complexity = complexity_map.get(query_metadata['complexity'], 0.5)
                example_complexity = complexity_map.get(metadata['complexity'], 0.5)
                complexity_diff = abs(query_complexity - example_complexity)
                complexity_score = 1.0 - complexity_diff
            
            # Calculate final score with weights
            weights = {
                'similarity': 0.3,
                'step_back': 0.2,
                'component_match': 0.2,
                'metadata_match': 0.2,
                'complexity': 0.1
            }
            
            final_score = (
                weights['similarity'] * base_similarity +
                weights['step_back'] * (sum(step_back_scores.values()) / len(step_back_scores)) +
                weights['component_match'] * component_match +
                weights['metadata_match'] * metadata_match +
                weights['complexity'] * complexity_score
            )
            
            ranked_results.append({
                'description': doc,
                'metadata': metadata,
                'scores': {
                    'final': final_score,
                    'similarity': base_similarity,
                    'step_back': step_back_scores,
                    'component_match': component_match,
                    'metadata_match': metadata_match,
                    'complexity': complexity_score
                }
            })
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['scores']['final'], reverse=True)
        return ranked_results

    def extract_metadata(self, description: str, code: str) -> dict:
        """Extract metadata from a description and code using LLM"""
        try:
            # Use MetadataExtractor for initial metadata
            metadata = self.metadata_extractor.extract_metadata(description)
            
            # Add code-based metadata
            code_metadata = self._analyze_code_metadata(code)
            metadata.update(code_metadata)
            
            # Log the extraction process
            print("\nMetadata Extraction Results:")
            print("- From description:", json.dumps(metadata, indent=2))
            print("- From code analysis:", json.dumps(code_metadata, indent=2))
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return {}
    
    def analyze_categories(self, description: str, code: str) -> dict:
        """Analyze and categorize the object using standardized categories"""
        try:
            # First get the object type from metadata
            metadata = self.metadata_extractor.extract_metadata(description)
            object_type = metadata.get('object_type', 'unknown')
            
            # Format the categories and properties info
            categories_info = "\n".join(
                f"- {cat}: {info['description']} (e.g., {', '.join(info['examples'])})"
                for cat, info in self.STANDARD_CATEGORIES.items()
            )
            
            properties_info = "\n".join(
                f"- {prop}: {', '.join(values)}"
                for prop, values in self.STANDARD_PROPERTIES.items()
            )
            
            # Use the category analysis prompt
            prompt = CATEGORY_ANALYSIS_PROMPT.format(
                object_type=object_type,
                description=description,
                categories_info=categories_info,
                properties_info=properties_info
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Parse the JSON response
            try:
                # Clean up the response to ensure it's valid JSON
                json_str = response.content.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("```", 1)[0]
                json_str = json_str.strip()
                
                result = json.loads(json_str)
                
                # Validate and clean the response
                cleaned_result = {
                    "categories": [
                        cat for cat in result.get("categories", [])
                        if cat in self.STANDARD_CATEGORIES
                    ],
                    "properties": {
                        prop: value
                        for prop, value in result.get("properties", {}).items()
                        if prop in self.STANDARD_PROPERTIES
                        and value in self.STANDARD_PROPERTIES[prop]
                    },
                    "similar_objects": [
                        obj for obj in result.get("similar_objects", [])
                        if any(obj in cat_info["examples"]
                              for cat_info in self.STANDARD_CATEGORIES.values())
                    ]
                }
                
                # Ensure at least one category is assigned
                if not cleaned_result["categories"]:
                    cleaned_result["categories"] = ["other"]
                
                # Log the analysis results
                print("\nCategory Analysis Results:")
                print(json.dumps(cleaned_result, indent=2))
                
                return cleaned_result
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {str(e)}")
                print(f"Raw response: {json_str}")
                return {
                    "categories": ["other"],
                    "properties": {},
                    "similar_objects": []
                }
                
        except Exception as e:
            print(f"Error analyzing categories: {str(e)}")
            return {
                "categories": ["other"],
                "properties": {},
                "similar_objects": []
            }
    
    def _analyze_code_metadata(self, code: str) -> dict:
        """Analyze OpenSCAD code to extract additional metadata"""
        metadata = {}
        
        try:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(code, {})
            if complexity_score < 30:
                metadata["complexity"] = "SIMPLE"
            elif complexity_score < 70:
                metadata["complexity"] = "MEDIUM"
            else:
                metadata["complexity"] = "COMPLEX"
            
            # Analyze components
            components = self._analyze_components(code)
            metadata["components"] = components
            
            # Extract geometric properties
            geometric_props = []
            if any(c["name"] == "sphere" for c in components):
                geometric_props.append("spherical")
            if any(c["name"] == "cube" for c in components):
                geometric_props.append("angular")
            if any(c["name"] == "cylinder" for c in components):
                geometric_props.append("cylindrical")
            if any(c["type"] == "boolean" for c in components):
                geometric_props.append("compound")
            metadata["geometric_properties"] = geometric_props
            
            # Analyze technical requirements
            tech_reqs = []
            if len([c for c in components if c["type"] == "boolean"]) > 2:
                tech_reqs.append("complex_boolean_operations")
            if len([c for c in components if c["type"] == "transformation"]) > 3:
                tech_reqs.append("multiple_transformations")
            if any(c["type"] == "module" for c in components):
                tech_reqs.append("modular_design")
            metadata["technical_requirements"] = tech_reqs
            
        except Exception as e:
            print(f"Error analyzing code metadata: {str(e)}")
        
        return metadata 

    def _fuzzy_match(self, str1, str2, threshold=0.8):
        """Compare two strings for fuzzy matching"""
        if not str1 or not str2:
            return False
            
        str1 = str(str1).lower()
        str2 = str(str2).lower()
        
        # Direct match
        if str1 == str2:
            return True
            
        # Substring match
        if str1 in str2 or str2 in str1:
            return True
            
        # Word overlap match
        words1 = set(str1.split())
        words2 = set(str2.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        if total == 0:
            return False
            
        similarity = overlap / total
        return similarity >= threshold 

    def _example_exists(self, example_id):
        """Check if an example with the given ID already exists"""
        try:
            existing = self.collection.get(ids=[example_id])
            return bool(existing and existing['ids'])
        except Exception:
            return False 