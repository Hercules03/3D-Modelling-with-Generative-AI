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
from typing import Dict, List
import logging
import traceback
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

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
            
            # Ensure metadata has required fields
            metadata.setdefault('features', [])
            metadata.setdefault('geometric_properties', [])
            metadata.setdefault('materials', [])
            metadata.setdefault('technical_requirements', [])
            metadata.setdefault('complexity', 'SIMPLE')
            metadata.setdefault('style', 'Modern')
            metadata.setdefault('use_case', [])
            
            # Convert string lists to actual lists if needed
            for field in ['features', 'geometric_properties', 'materials', 'technical_requirements', 'use_case']:
                if isinstance(metadata[field], str):
                    metadata[field] = [item.strip() for item in metadata[field].split(',') if item.strip()]
            
            # Perform step-back analysis
            step_back_prompt = STEP_BACK_PROMPT_TEMPLATE.format(query=description)
            step_back_response = self.llm.invoke(step_back_prompt)
            technical_analysis = step_back_response.content
            
            # Parse step-back analysis
            principles = []
            shape_components = []
            implementation_steps = []
            
            current_section = None
            for line in technical_analysis.split('\n'):
                line = line.strip()
                if 'CORE PRINCIPLES:' in line:
                    current_section = 'principles'
                elif 'SHAPE COMPONENTS:' in line:
                    current_section = 'shape_components'
                elif 'IMPLEMENTATION STEPS:' in line:
                    current_section = 'implementation_steps'
                elif line and line[0] in ['-', '•', '*'] and current_section == 'principles':
                    principles.append(line[1:].strip())
                elif line and line[0] in ['-', '•', '*'] and current_section == 'shape_components':
                    shape_components.append(line[1:].strip())
                elif line and (line[0].isdigit() or line[0] in ['-', '•', '*']) and current_section == 'implementation_steps':
                    implementation_steps.append(line[line.find('.')+1:].strip() if line[0].isdigit() else line[1:].strip())
            
            # Add step-back analysis to metadata
            metadata['step_back_analysis'] = {
                'core_principles': principles,
                'shape_components': shape_components,
                'implementation_steps': implementation_steps
            }
            
            # Flatten metadata for ChromaDB while preserving list and dict structures
            flattened_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # For dictionaries, store as JSON string
                    flattened_metadata[key] = json.dumps(value)
                elif isinstance(value, list):
                    # For lists, store as JSON string
                    flattened_metadata[key] = json.dumps(value)
                else:
                    # For scalar values, convert to string
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
            traceback.print_exc()
            return False
    
    def get_relevant_examples(self, query: str, similarity_threshold: float = 0.15) -> List[Dict]:
        """
        Get relevant examples based on the query.
        
        Args:
            query: The search query
            similarity_threshold: Minimum similarity score required (default: 0.15)
            
        Returns:
            List of relevant examples with their similarity scores
        """
        try:
            # Extract metadata from query
            query_metadata = self.metadata_extractor.extract_metadata(query)
            
            # Query the vector store
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            if not results or not results['ids']:
                print("No results found in vector store")
                return []
            
            # Prepare results for ranking
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                search_results.append(result)
            
            print("\nRaw Results:")
            for i, result in enumerate(search_results, 1):
                print(f"\nRaw Result {i}:")
                print(f"ID: {result['id']}")
                print(f"Distance: {result['distance']}")
                print(f"Metadata: {result['metadata']}")
            
            # Rank results
            ranked_results = self._rank_results(query_metadata, search_results)
            
            print("\nScores before filtering:")
            for result in ranked_results:
                print("\nExample: ")
                print(f"Final Score: {result['score']:.3f}")
                print("Component Scores:")
                for name, score in result['score_breakdown']['component_scores'].items():
                    print(f"  {name}: {score:.3f}")
                print("Step-back Details:")
                for name, score in result['score_breakdown']['step_back_details'].items():
                    print(f"  {name}: {score:.3f}")
            
            # Filter by similarity threshold
            relevant_results = [
                result for result in ranked_results
                if result['score'] >= similarity_threshold
            ]
            
            print(f"\nFound {len(relevant_results)} relevant examples (threshold: {similarity_threshold}):")
            for result in relevant_results:
                print(f"\nExample (Score: {result['score']:.3f}):")
                print(f"ID: {result['example']['id']}")
                print(f"Metadata: {result['example']['metadata']}")
                print("Score Breakdown:")
                for name, score in result['score_breakdown']['component_scores'].items():
                    print(f"  {name}: {score:.3f}")
                
            return relevant_results
            
        except Exception as e:
            print(f"Error getting relevant examples: {str(e)}")
            traceback.print_exc()
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

    def _rank_results(self, query_metadata, results):
        """
        Rank and filter search results based on metadata similarity.
        """
        ranked_results = []
        
        for result in results:
            try:
                result_metadata = result['metadata']
                
                # Convert string fields to lists if they're strings
                for field in ['features', 'geometric_properties']:
                    if isinstance(result_metadata.get(field, ''), str):
                        result_metadata[field] = [x.strip() for x in result_metadata[field].split(',')]
                    if isinstance(query_metadata.get(field, ''), str):
                        query_metadata[field] = [x.strip() for x in query_metadata[field].split(',')]

                # Parse step-back analysis fields
                try:
                    # Handle old format with separate fields
                    if 'step_back_analysis_principles' in result_metadata:
                        result_metadata['step_back_analysis'] = {
                            'core_principles': json.loads(result_metadata['step_back_analysis_principles']),
                            'shape_components': json.loads(result_metadata.get('step_back_analysis_abstractions', '[]')),
                            'implementation_steps': json.loads(result_metadata.get('step_back_analysis_approach', '[]'))
                        }
                    elif isinstance(result_metadata.get('step_back_analysis', ''), str):
                        result_metadata['step_back_analysis'] = json.loads(result_metadata['step_back_analysis'])
                    
                    if isinstance(query_metadata.get('step_back_analysis', ''), str):
                        query_metadata['step_back_analysis'] = json.loads(query_metadata['step_back_analysis'])
                except:
                    result_metadata['step_back_analysis'] = {
                        'core_principles': [],
                        'shape_components': [],
                        'implementation_steps': []
                    }

                # Calculate component match score
                result_components = result_metadata.get('step_back_analysis', {}).get('shape_components', [])
                query_components = query_metadata.get('step_back_analysis', {}).get('shape_components', [])
                component_match = self._calculate_text_similarity(query_components, result_components)

                # Calculate step-back analysis score
                step_back_score = 0
                if 'step_back_analysis' in result_metadata and 'step_back_analysis' in query_metadata:
                    principles_score = self._calculate_text_similarity(
                        query_metadata['step_back_analysis'].get('core_principles', []),
                        result_metadata['step_back_analysis'].get('core_principles', [])
                    )
                    components_score = self._calculate_text_similarity(
                        query_metadata['step_back_analysis'].get('shape_components', []),
                        result_metadata['step_back_analysis'].get('shape_components', [])
                    )
                    steps_score = self._calculate_text_similarity(
                        query_metadata['step_back_analysis'].get('implementation_steps', []),
                        result_metadata['step_back_analysis'].get('implementation_steps', [])
                    )
                    step_back_score = (principles_score + components_score + steps_score) / 3

                # Calculate geometric properties match
                geometric_score = self._calculate_text_similarity(
                    query_metadata.get('geometric_properties', []),
                    result_metadata.get('geometric_properties', [])
                )

                # Calculate feature match
                feature_score = self._calculate_text_similarity(
                    query_metadata.get('features', []),
                    result_metadata.get('features', [])
                )

                # Calculate style match (case-insensitive and fuzzy)
                query_style = str(query_metadata.get('style', '')).lower()
                result_style = str(result_metadata.get('style', '')).lower()
                style_score = fuzz.ratio(query_style, result_style) / 100.0

                # Calculate complexity match (case-insensitive)
                query_complexity = str(query_metadata.get('complexity', '')).upper()
                result_complexity = str(result_metadata.get('complexity', '')).upper()
                complexity_score = 1.0 if query_complexity == result_complexity else 0.0

                # Calculate final score with weights
                weights = {
                    'component_match': 0.25,
                    'step_back_match': 0.2,
                    'geometric_match': 0.15,
                    'feature_match': 0.2,
                    'style_match': 0.1,
                    'complexity_match': 0.1
                }

                final_score = (
                    weights['component_match'] * component_match +
                    weights['step_back_match'] * step_back_score +
                    weights['geometric_match'] * geometric_score +
                    weights['feature_match'] * feature_score +
                    weights['style_match'] * style_score +
                    weights['complexity_match'] * complexity_score
                )

                # Create score breakdown
                score_breakdown = {
                    'final_score': final_score,
                    'component_scores': {
                        'component_match': component_match,
                        'step_back_match': step_back_score,
                        'geometric_match': geometric_score,
                        'feature_match': feature_score,
                        'style_match': style_score,
                        'complexity_match': complexity_score
                    },
                    'step_back_details': {
                        'principles': principles_score,
                        'abstractions': components_score,
                        'approach': steps_score
                    }
                }

                ranked_results.append({
                    'example': result,
                    'score': final_score,
                    'score_breakdown': score_breakdown
                })

            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue

        # Sort by score in descending order
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results

    def _calculate_text_similarity(self, list1, list2):
        """
        Calculate similarity between two lists of text items using fuzzy matching.
        """
        if not list1 or not list2:
            return 0.0
        
        total_score = 0
        max_scores = []
        
        for item1 in list1:
            item_scores = []
            for item2 in list2:
                ratio = fuzz.ratio(str(item1).lower(), str(item2).lower()) / 100.0
                item_scores.append(ratio)
            max_scores.append(max(item_scores))
        
        if max_scores:
            total_score = sum(max_scores) / len(max_scores)
        
        return total_score

    def _validate_metadata(self, metadata):
        """Validate the metadata structure."""
        required_fields = [
            'object_type',
            'dimensions',
            'features',
            'materials',
            'complexity',
            'style',
            'use_case',
            'geometric_properties',
            'technical_requirements',
            'step_back_analysis'
        ]

        # Check top-level required fields
        for field in required_fields:
            if field not in metadata:
                print(f"Missing required field: {field}")
                return False

        # Check step-back analysis structure
        step_back_fields = [
            'core_principles',  # Updated from 'principles'
            'shape_components',
            'implementation_steps'
        ]

        for field in step_back_fields:
            if field not in metadata['step_back_analysis']:
                print(f"Missing required step-back analysis field: {field}")
                return False

        return True

    def _extract_metadata(self, description: str, code: str = "") -> Dict:
        """Extract metadata from description and code"""
        try:
            # Get metadata from extractor
            metadata = self.metadata_extractor.extract_metadata(description, code)
            if not metadata:
                logger.error("Failed to extract metadata")
                return None
            
            # Ensure required fields exist with default values if missing
            metadata.setdefault('materials', [])
            metadata.setdefault('dimensions', {})
            metadata.setdefault('features', [])
            metadata.setdefault('use_case', [])
            metadata.setdefault('geometric_properties', [])
            metadata.setdefault('technical_requirements', [])
            
            # Ensure step-back analysis exists
            if 'step_back_analysis' not in metadata:
                metadata['step_back_analysis'] = {
                    'principles': [],
                    'abstractions': [],
                    'approach': []
                }
            
            # Validate metadata
            if not self._validate_metadata(metadata):
                logger.error("Invalid metadata structure")
                return None
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None
    
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

    def _calculate_component_match(self, query_components, result_components):
        """Calculate similarity score based on OpenSCAD component matching"""
        total_score = 0.0
        for query_item in query_components:
            best_match = max(
                (self._fuzzy_match(query_item['name'], result_item['name'])
                 for result_item in result_components),
                default=0.0
            )
            if best_match > 0.6:  # Threshold for considering it a match
                total_score += best_match
        
        return total_score / max(len(query_components), len(result_components)) if query_components else 0.0

    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Calculate fuzzy match similarity between two strings"""
        # Remove punctuation and convert to lowercase
        str1 = ''.join(c.lower() for c in str1 if c.isalnum() or c.isspace())
        str2 = ''.join(c.lower() for c in str2 if c.isalnum() or c.isspace())
        
        # Split into words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        # Calculate Jaccard similarity for word sets
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union

    def _example_exists(self, example_id):
        """Check if an example with the given ID already exists"""
        try:
            existing = self.collection.get(ids=[example_id])
            return bool(existing and existing['ids'])
        except Exception:
            return False

    def _group_components_by_type(self, components: List[Dict]) -> Dict[str, List[Dict]]:
        """Group components by their type"""
        grouped = {}
        for component in components:
            if isinstance(component, dict) and 'type' in component and 'name' in component:
                comp_type = component['type'].lower()
                if comp_type not in grouped:
                    grouped[comp_type] = []
                grouped[comp_type].append(component)
        return grouped 