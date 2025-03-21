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
                import hashlib
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

    def add_example(self, description, code):
        """Add a new example to the knowledge base with metadata"""
        try:
            # Generate unique ID based on content hash
            import hashlib
            content_hash = hashlib.md5(f"{description}{code}".encode()).hexdigest()[:8]
            example_id = f"{self._generate_base_name(description)}_{content_hash}"
            
            # Check if example already exists
            try:
                existing = self.collection.get(ids=[example_id])
                if existing and existing['ids']:
                    print(f"\nSkipping duplicate example with ID: {example_id}")
                    print("This example is already in the knowledge base.")
                    return True
            except Exception:
                pass  # Example doesn't exist, continue with adding it
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(description)
            
            # Use LLM for dynamic categorization if object_type is present
            if "object_type" in metadata:
                object_type = metadata["object_type"].lower()
                category_analysis = self._analyze_object_categories(object_type, description)
                
                if category_analysis:
                    # Add categories from LLM analysis
                    metadata["categories"] = category_analysis.get("categories", ["other"])
                    
                    # Add properties from LLM analysis
                    llm_properties = category_analysis.get("properties", {})
                    metadata["properties"] = {
                        **metadata.get("properties", {}),
                        **llm_properties
                    }
                    
                    # Add similar objects for better matching
                    metadata["similar_objects"] = category_analysis.get("similar_objects", [])
            
            # Flatten nested structures in metadata
            flattened_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # For dictionaries, create separate keys with prefixes
                    for sub_key, sub_value in value.items():
                        flattened_metadata[f"{key}_{sub_key}"] = str(sub_value)
                elif isinstance(value, list):
                    # For lists, join into a comma-separated string
                    flattened_metadata[key] = ", ".join(str(item) for item in value)
                else:
                    flattened_metadata[key] = str(value)
            
            # Add to ChromaDB with flattened metadata
            timestamp = datetime.now().isoformat()
            self.collection.add(
                documents=[description],
                metadatas=[{
                    "code": code,
                    "timestamp": timestamp,
                    "type": "scad_generation",
                    "user_accepted": True,
                    **flattened_metadata  # Include flattened metadata
                }],
                ids=[example_id]
            )
            
            # Update last processed timestamp
            self.last_processed_time = timestamp
            self._save_last_processed_time()
            
            print(f"\nExample added successfully with ID: {example_id}")
            print("Stored metadata:")
            for key, value in flattened_metadata.items():
                print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"\nError adding example: {e}")
            return False
    
    def get_relevant_examples(self, query, max_examples=2, filters=None, similarity_threshold=0.7):
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
            else:
                print(f"\nNo examples met the similarity threshold of {similarity_threshold}")
            
            return filtered_results
            
        except Exception as e:
            print(f"Error getting relevant examples: {str(e)}")
            return []
            
    def _extract_object_type(self, description):
        """Extract the main object type from a description"""
        try:
            response = self.llm.invoke(self.keyword_prompt.format(description=description))
            return response.content.strip().lower()
        except Exception as e:
            print(f"Error extracting object type: {str(e)}")
            return "unknown"
    
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
        
        try:
            print("\nDEBUG: Results structure:")
            print(f"Keys in results: {results.keys()}")
            print(f"First document: {results['documents'][0][0] if results['documents'][0] else 'None'}")
            print(f"First metadata: {results['metadatas'][0][0] if results['metadatas'][0] else 'None'}")
            print(f"First distance: {results['distances'][0][0] if results['distances'][0] else 'None'}")
            
            for i, (desc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"\nProcessing result {i}:")
                print(f"Description: {desc[:100]}...")
                
                # Get code from metadata
                code = metadata.get('code', '')
                if not code:
                    print(f"No code found in metadata")
                    print(f"Available metadata keys: {metadata.keys()}")
                    continue
                
                # Calculate base similarity score (convert distance to similarity)
                similarity = 1 - min(distance, 1.0)  # Ensure similarity is between 0 and 1
                print(f"Base similarity score: {similarity:.3f}")
                
                # Calculate step-back analysis similarity scores
                step_back_scores = {'principles': 0.0, 'abstractions': 0.0, 'approach': 0.0}
                
                if 'step_back_analysis' in query_metadata:
                    query_analysis = query_metadata['step_back_analysis']
                    example_analysis = None
                    
                    # Try to extract step-back analysis from example description
                    try:
                        analysis_start = desc.find('Technical Analysis:')
                        if analysis_start != -1:
                            analysis_text = desc[analysis_start:]
                            
                            # Parse example's step-back analysis
                            example_principles = []
                            example_abstractions = []
                            example_approach = []
                            
                            current_section = None
                            for line in analysis_text.split('\n'):
                                line = line.strip()
                                if 'CORE PRINCIPLES:' in line:
                                    current_section = 'principles'
                                elif 'SHAPE COMPONENTS:' in line:
                                    current_section = 'abstractions'
                                elif 'IMPLEMENTATION STEPS:' in line:
                                    current_section = 'approach'
                                elif line and line[0] == '-' and current_section == 'principles':
                                    example_principles.append(line[1:].strip())
                                elif line and line[0] == '-' and current_section == 'abstractions':
                                    example_abstractions.append(line[1:].strip())
                                elif line and line[0].isdigit() and current_section == 'approach':
                                    example_approach.append(line[line.find('.')+1:].strip())
                            
                            example_analysis = {
                                'principles': example_principles,
                                'abstractions': example_abstractions,
                                'approach': example_approach
                            }
                    except Exception as e:
                        print(f"Error parsing example step-back analysis: {str(e)}")
                    
                    if example_analysis:
                        # Calculate similarity scores for each section
                        for section in ['principles', 'abstractions', 'approach']:
                            query_items = query_analysis[section]
                            example_items = example_analysis[section]
                            
                            matching_items = 0
                            total_items = len(query_items)
                            
                            for query_item in query_items:
                                query_words = set(query_item.lower().split())
                                for example_item in example_items:
                                    example_words = set(example_item.lower().split())
                                    # Calculate word overlap
                                    overlap = len(query_words & example_words) / len(query_words | example_words)
                                    if overlap > 0.3:  # If more than 30% words match
                                        matching_items += overlap
                            
                            step_back_scores[section] = matching_items / total_items if total_items > 0 else 0
                            print(f"{section.capitalize()} similarity: {step_back_scores[section]:.3f}")
                
                # Calculate complexity score
                complexity_score = self._calculate_complexity_score(code, metadata)
                print(f"Complexity score: {complexity_score}")
                
                # Calculate component match score
                query_components = query_metadata.get('components', [])
                example_components = self._analyze_components(code)
                component_match = 0
                
                # Handle both dictionary and string formats for components
                if query_components:
                    # If query_components is a dictionary with 'components' key
                    if isinstance(query_components, dict) and 'components' in query_components:
                        query_components = query_components['components']
                    
                    # Count matching components
                    matching_components = 0
                    total_components = len(query_components)
                    for qc in query_components:
                        qc_name = qc['name'] if isinstance(qc, dict) else qc
                        for ec in example_components:
                            ec_name = ec['name'] if isinstance(ec, dict) else ec
                            if qc_name.lower() == ec_name.lower():
                                matching_components += 1
                                break
                
                    component_match = matching_components / total_components if total_components > 0 else 0
                    print(f"Component match score: {component_match:.3f} ({matching_components}/{total_components} components)")
                
                # Calculate metadata match score
                metadata_match = 0
                matching_props = 0
                total_props = 0
                
                if query_metadata and metadata:
                    # Important properties to check
                    key_properties = ['object_type', 'complexity', 'style', 'use_case']
                    
                    for k in key_properties:
                        if k in query_metadata and k in metadata:
                            total_props += 1
                            if str(metadata[k]).lower() == str(query_metadata[k]).lower():
                                matching_props += 1
                
                    # Check arrays (like geometric_properties, features)
                    array_properties = ['geometric_properties', 'features']
                    for k in array_properties:
                        if k in query_metadata and k in metadata:
                            query_values = query_metadata[k] if isinstance(query_metadata[k], list) else [query_metadata[k]]
                            example_values = metadata[k].split(', ') if isinstance(metadata[k], str) else metadata[k]
                            
                            total_props += len(query_values)
                            for qv in query_values:
                                if any(str(qv).lower() in str(ev).lower() for ev in example_values):
                                    matching_props += 1
                
                    metadata_match = matching_props / total_props if total_props > 0 else 0
                    print(f"Metadata match score: {metadata_match:.3f} ({matching_props}/{total_props} properties)")
                
                # Calculate step-back analysis score (average of all sections)
                step_back_score = sum(step_back_scores.values()) / len(step_back_scores)
                print(f"Step-back analysis score: {step_back_score:.3f}")
                
                # Calculate final score with adjusted weights
                final_score = (
                    similarity * 0.25 +           # Base similarity (25%)
                    step_back_score * 0.25 +      # Step-back analysis matching (25%)
                    component_match * 0.25 +      # Component matching (25%)
                    metadata_match * 0.15 +       # Metadata matching (15%)
                    (complexity_score/100) * 0.1  # Complexity score (10%)
                )
                
                print(f"Final score: {final_score:.3f}")
                
                ranked_results.append({
                    'description': desc,
                    'metadata': metadata,
                    'code': code,
                    'scores': {
                        'final': final_score,
                        'similarity': similarity,
                        'step_back': step_back_scores,
                        'component_match': component_match,
                        'metadata_match': metadata_match,
                        'complexity': complexity_score
                    }
                })
                
        except Exception as e:
            print(f"Error ranking results: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x['scores']['final'], reverse=True)
        return ranked_results 