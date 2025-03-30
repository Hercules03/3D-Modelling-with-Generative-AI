import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from constant import *
from datetime import datetime

import hashlib
from typing import Dict, List
import logging
import traceback
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class SCADKnowledgeBase:
    logger.info("Working on SCADKnowledgeBase Class")
    def __init__(self, keyword_extractor, metadata_extractor, conversation_logger):
        """Initialize the enhanced knowledge base with ChromaDB
        
        Args:
            load_new_examples_from_logs: Whether to check for new examples in conversation logs.
                                         Defaults to False as examples are now directly added to
                                         ChromaDB when they are approved by the user.
        """
        self.debug_log = []
        logger.info("Initializing SCAD Knowledge Base...")
        
        # Set up logger for conversation logs
        self.logger = conversation_logger
        
        # Set up paths
        print("Setting up paths...")
        self.persistence_dir = os.path.join(os.getcwd(), "scad_knowledge_base/chroma")
        logger.info(f"Persistence directory: {self.persistence_dir}")
        
        self.keyword_extractor = keyword_extractor
        self.metadata_extractor = metadata_extractor
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=self.persistence_dir)
            print("- ChromaDB client initialized")
            logger.info("ChromaDB client initialized")
            
            # Initialize embedding function
            self.embedding_function = SentenceTransformerEmbeddingFunction()
            print("- Sentence transformer embedding function initialized")
            logger.info("Sentence transformer embedding function initialized")
            # Get or create collection
            try:
                logger.info("Getting or creating collection...")
                self.collection = self.client.get_collection(
                    name="scad_examples",
                    embedding_function=self.embedding_function
                )
                example_count = self.collection.count()
                print(f"- Using existing collection 'scad_examples' with {example_count} examples")
                logger.info(f"- Using existing collection 'scad_examples' with {example_count} examples")
            except ValueError:
                logger.info("- Creating new collection 'scad_examples'...")
                self.collection = self.client.create_collection(
                    name="scad_examples",
                    embedding_function=self.embedding_function
                )
                print("- New collection created successfully")
                logger.info("- New collection created successfully")
                example_count = 0
            
            print("\n" + "="*50)
            print(f"Knowledge Base Ready | {example_count} examples loaded")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"\nError initializing ChromaDB: {str(e)}")
            raise
        
    def write_debug(self, *messages):
        """Write messages to debug log"""
        for message in messages:
            self.debug_log.append(message)
    
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
    
    def _save_last_processed_time(self):
        """Save the timestamp of last processed example"""
        os.makedirs(self.persistence_dir, exist_ok=True)
        with open(f"{self.persistence_dir}/last_processed.json", "w") as f:
            json.dump({"timestamp": self.last_processed_time}, f)
    
    
    def _analyze_object_categories(self, object_type, description):
        """Use LLM to analyze object categories using standardized categories and properties"""
        # Use the centralized method from MetadataExtractor
        metadata = {'object_type': object_type}
        return self.metadata_extractor.analyze_categories(description, metadata)

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
                print("Warning: No metadata provided, extracting from description...")
                metadata = self.metadata_extractor.extract_metadata(description)
            
            # Ensure metadata has required fields
            metadata.setdefault('features', [])
            metadata.setdefault('geometric_properties', [])
            metadata.setdefault('materials', [])
            metadata.setdefault('technical_requirements', [])
            metadata.setdefault('complexity', 'SIMPLE')
            metadata.setdefault('style', 'Modern')
            metadata.setdefault('use_case', [])
            
            # Serialize all list and dictionary values to JSON strings
            serialized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    serialized_metadata[key] = json.dumps(value)
                else:
                    serialized_metadata[key] = value
            
            # Add code and other metadata fields
            serialized_metadata['code'] = code
            serialized_metadata['timestamp'] = datetime.now().isoformat()
            serialized_metadata['type'] = 'scad_generation'
            serialized_metadata['user_accepted'] = True
            serialized_metadata['description'] = description
            
            # Add example to ChromaDB
            print(f"\nAdding example with ID: {example_id}")
            self.collection.add(
                documents=[description],
                metadatas=[serialized_metadata],
                ids=[example_id]
            )
            
            # Update and save last processed timestamp
            current_time = datetime.now().isoformat()
            self.last_processed_time = current_time
            self._save_last_processed_time()
            print(f"- Updated last processed time to {current_time}")
            
            return True
            
        except Exception as e:
            print(f"Error adding example: {str(e)}")
            traceback.print_exc()
            return False
            
    def _generate_base_name(self, description):
        """Generate a base name for the example"""
        # First try to get a meaningful name using keyword extraction
        try:
            keyword_data = self.keyword_extractor.extract_keyword(description)
            if keyword_data and keyword_data.get('core_type'):
                return ''.join(c for c in keyword_data['core_type'] if c.isalnum())
        except Exception as e:
            print(f"Warning: Keyword extraction failed for base name generation: {e}")
        
        # Fall back to simple word filtering if keyword extraction fails
        stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she'
        }
        
        # Convert to lowercase and split into words
        words = description.lower().split()
        
        # Find first meaningful word
        for word in words:
            if word not in stop_words:
                return ''.join(c for c in word if c.isalnum())
        
        return 'example'

    def _extract_metadata(self, description: str, code: str = "") -> Dict:
        """Extract metadata from description and code using MetadataExtractor"""
        try:
            # Get metadata from extractor
            metadata = self.metadata_extractor.extract_metadata(
                description=description,
                code=code,
                step_back_result=None,  # Will be added if needed
                keyword_data=self.keyword_extractor.extract_keyword(description)
            )
            
            if not metadata:
                logger.error("Failed to extract metadata")
                return None
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def analyze_categories(self, description: str, code: str) -> dict:
        """Analyze and categorize the object using MetadataExtractor"""
        try:
            # Get metadata first to ensure we have the object type
            metadata = self.metadata_extractor.extract_metadata(description, code)
            return self.metadata_extractor.analyze_categories(description, metadata)
                
        except Exception as e:
            print(f"Error analyzing categories: {str(e)}")
            return {
                "categories": ["other"],
                "properties": {},
                "similar_objects": []
            }

    def _analyze_code_metadata(self, code: str) -> dict:
        """Analyze OpenSCAD code to extract additional metadata using MetadataExtractor"""
        return self.metadata_extractor.analyze_code_metadata(code)

    def _calculate_complexity_score(self, code, metadata):
        """Calculate complexity score based on code analysis and metadata"""
        # Use the centralized method from MetadataExtractor
        return self.metadata_extractor.calculate_complexity_with_metadata(code, metadata)

    def _analyze_components(self, code):
        """Analyze and extract components from SCAD code"""
        # Use the centralized method from MetadataExtractor
        return self.metadata_extractor.analyze_components(code)

    def _rank_results(self, query_metadata, results):
        """
        Rank and filter search results based on metadata similarity.
        """
        ranked_results = []
        
        # Log the start of the ranking process with detailed info
        self.write_debug(
            "\n=== RANKING PROCESS DETAILS ===\n",
            f"Number of results to rank: {len(results)}\n",
            f"Query metadata: {json.dumps(query_metadata, default=str)}\n",
            "=" * 50 + "\n"
        )
        
        print("\nRe-ranking Process Details:")
        print("=" * 50)
        print(f"Number of results to rank: {len(results)}")
        
        for result in results:
            try:
                result_metadata = result['metadata']
                
                # Ensure all required fields exist with default values
                result_metadata.setdefault('features', [])
                result_metadata.setdefault('geometric_properties', [])
                result_metadata.setdefault('step_back_analysis', {})
                result_metadata.setdefault('style', 'Modern')
                result_metadata.setdefault('complexity', 'SIMPLE')
                result_metadata.setdefault('object_type', '')
                
                query_metadata.setdefault('features', [])
                query_metadata.setdefault('geometric_properties', [])
                query_metadata.setdefault('step_back_analysis', {})
                query_metadata.setdefault('style', 'Modern')
                query_metadata.setdefault('complexity', 'SIMPLE')
                query_metadata.setdefault('object_type', '')

                # Convert string fields to lists if they're strings
                for field in ['features', 'geometric_properties']:
                    if isinstance(result_metadata.get(field), str):
                        try:
                            result_metadata[field] = json.loads(result_metadata[field])
                        except:
                            result_metadata[field] = [x.strip() for x in result_metadata[field].split(',') if x.strip()]
                    if isinstance(query_metadata.get(field), str):
                        try:
                            query_metadata[field] = json.loads(query_metadata[field])
                        except:
                            query_metadata[field] = [x.strip() for x in query_metadata[field].split(',') if x.strip()]

                # Parse step-back analysis fields
                try:
                    if 'step_back_analysis_principles' in result_metadata:
                        result_metadata['step_back_analysis'] = {
                            'core_principles': json.loads(result_metadata['step_back_analysis_principles']),
                            'shape_components': json.loads(result_metadata.get('step_back_analysis_abstractions', '[]')),
                            'implementation_steps': json.loads(result_metadata.get('step_back_analysis_approach', '[]'))
                        }
                    elif isinstance(result_metadata.get('step_back_analysis'), str):
                        result_metadata['step_back_analysis'] = json.loads(result_metadata['step_back_analysis'])
                    
                    if isinstance(query_metadata.get('step_back_analysis'), str):
                        query_metadata['step_back_analysis'] = json.loads(query_metadata['step_back_analysis'])
                except Exception as e:
                    print(f"Error parsing step-back analysis: {str(e)}")
                    result_metadata['step_back_analysis'] = {
                        'core_principles': [],
                        'shape_components': [],
                        'implementation_steps': []
                    }

                # Ensure step-back analysis structure
                for metadata in [result_metadata, query_metadata]:
                    metadata['step_back_analysis'] = metadata.get('step_back_analysis', {})
                    if isinstance(metadata['step_back_analysis'], str):
                        try:
                            metadata['step_back_analysis'] = json.loads(metadata['step_back_analysis'])
                        except:
                            metadata['step_back_analysis'] = {}
                    
                    metadata['step_back_analysis'].setdefault('core_principles', [])
                    metadata['step_back_analysis'].setdefault('shape_components', [])
                    metadata['step_back_analysis'].setdefault('implementation_steps', [])

                # Debug step-back analysis
                print(f"\nResult ID: {result.get('id', 'unknown')}")
                print(f"Result object_type: {result_metadata.get('object_type', '')}")
                print(f"Result features: {result_metadata.get('features', [])}")
                print(f"Query object_type: {query_metadata.get('object_type', '')}")
                print(f"Query features: {query_metadata.get('features', [])}")

                # Calculate component match score
                result_components = result_metadata['step_back_analysis'].get('shape_components', [])
                query_components = query_metadata['step_back_analysis'].get('shape_components', [])
                component_match = self.metadata_extractor.calculate_text_similarity(query_components, result_components)

                # Calculate step-back analysis score
                principles_score = self.metadata_extractor.calculate_text_similarity(
                    query_metadata['step_back_analysis']['core_principles'],
                    result_metadata['step_back_analysis']['core_principles']
                )
                components_score = self.metadata_extractor.calculate_text_similarity(
                    query_metadata['step_back_analysis']['shape_components'],
                    result_metadata['step_back_analysis']['shape_components']
                )
                steps_score = self.metadata_extractor.calculate_text_similarity(
                    query_metadata['step_back_analysis']['implementation_steps'],
                    result_metadata['step_back_analysis']['implementation_steps']
                )
                step_back_score = (principles_score + components_score + steps_score) / 3

                # Calculate geometric properties match
                geometric_score = self.metadata_extractor.calculate_text_similarity(
                    query_metadata['geometric_properties'],
                    result_metadata['geometric_properties']
                )

                # Calculate feature match
                feature_score = self.metadata_extractor.calculate_text_similarity(
                    query_metadata['features'],
                    result_metadata['features']
                )

                # Calculate style match
                query_style = str(query_metadata['style']).lower()
                result_style = str(result_metadata['style']).lower()
                style_score = fuzz.ratio(query_style, result_style) / 100.0

                # Calculate complexity match
                query_complexity = str(query_metadata['complexity']).upper()
                result_complexity = str(result_metadata['complexity']).upper()
                complexity_score = 1.0 if query_complexity == result_complexity else 0.0
                
                # Calculate object type match
                query_object_type = str(query_metadata.get('object_type', '')).lower()
                result_object_type = str(result_metadata.get('object_type', '')).lower()
                object_type_score = fuzz.ratio(query_object_type, result_object_type) / 100.0

                # Calculate final score with weights
                weights = {
                    'component_match': 0.03,
                    'geometric_match': 0.03,
                    'step_back_match': 0.03,
                    'feature_match': 0.2,
                    'object_type_match': 0.7,
                    'style_match': 0.005,
                    'complexity_match': 0.005
                }

                final_score = (
                    weights['component_match'] * component_match +
                    weights['step_back_match'] * step_back_score +
                    weights['geometric_match'] * geometric_score +
                    weights['feature_match'] * feature_score +
                    weights['object_type_match'] * object_type_score +
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
                        'object_type_match': object_type_score,
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
                traceback.print_exc()
                continue

        # Sort by score in descending order
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\nFinal Ranking:")
        print("=" * 50)
        for i, result in enumerate(ranked_results, 1):
            print(f"\nRank {i}:")
            print(f"Example ID: {result['example'].get('id', 'unknown')}")
            print(f"Final Score: {result['score']:.3f}")
            print("Component Scores:")
            for name, score in result['score_breakdown']['component_scores'].items():
                print(f"- {name}: {score:.3f}")
        
        return ranked_results

    def _validate_metadata(self, metadata):
        """Validate the metadata structure."""
        # Use the centralized method from MetadataExtractor
        return self.metadata_extractor.validate_metadata(metadata)

    def _example_exists(self, example_id):
        """Check if an example with the given ID already exists"""
        try:
            existing = self.collection.get(ids=[example_id])
            return bool(existing and existing['ids'])
        except Exception:
            return False

    def _group_components_by_type(self, components: List[Dict]) -> Dict[str, List[Dict]]:
        """Group components by their type"""
        # Use the centralized method from MetadataExtractor
        return self.metadata_extractor._group_components_by_type(components)

    def delete_examples(self, example_ids: List[str]) -> bool:
        """Delete multiple examples from the knowledge base"""
        try:
            self.collection.delete(ids=example_ids)
            return True
        except Exception as e:
            print(f"Error deleting examples: {str(e)}")
            traceback.print_exc()
            return False

    def search_examples(self, search_term: str = None, filters: Dict = None, page: int = 1, page_size: int = 10) -> Dict:
        """
        Search for examples with pagination and filtering
        
        Args:
            search_term: Optional search term to filter descriptions
            filters: Optional dictionary of metadata filters
            page: Page number (1-based)
            page_size: Number of items per page
        """
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Get all examples first
            results = self.collection.get()
            if not results or not results['ids']:
                return {
                    'examples': [],
                    'total': 0,
                    'page': page,
                    'total_pages': 0
                }
            
            # Prepare examples list
            examples = []
            for i, (id, doc, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
                # Apply search term filter if provided
                if search_term and search_term.lower() not in doc.lower():
                    continue
                
                # Apply metadata filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in metadata:
                            # Handle JSON string values
                            meta_value = metadata[key]
                            if isinstance(meta_value, str):
                                if meta_value.startswith('[') or meta_value.startswith('{'):
                                    try:
                                        meta_value = json.loads(meta_value)
                                    except:
                                        pass
                            
                            # Check if value matches filter
                            if isinstance(meta_value, list):
                                if value not in meta_value:
                                    skip = True
                                    break
                            elif str(meta_value).lower() != str(value).lower():
                                skip = True
                                break
                    if skip:
                        continue
                
                # Parse metadata
                parsed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, str):
                        if value.startswith('[') or value.startswith('{'):
                            try:
                                parsed_metadata[key] = json.loads(value)
                            except:
                                parsed_metadata[key] = value
                        else:
                            parsed_metadata[key] = value
                    else:
                        parsed_metadata[key] = value
                
                examples.append({
                    'id': id,
                    'description': doc,
                    'metadata': parsed_metadata
                })
            
            # Calculate pagination
            total = len(examples)
            total_pages = (total + page_size - 1) // page_size
            
            # Apply pagination
            start = offset
            end = min(start + page_size, total)
            paginated_examples = examples[start:end]
            
            return {
                'examples': paginated_examples,
                'total': total,
                'page': page,
                'total_pages': total_pages
            }
            
        except Exception as e:
            print(f"Error searching examples: {str(e)}")
            traceback.print_exc()
            return {
                'examples': [],
                'total': 0,
                'page': page,
                'total_pages': 0
            }

    def get_example_details(self, example_id: str) -> Dict:
        """Get detailed information about a specific example"""
        try:
            result = self.collection.get(ids=[example_id])
            if not result or not result['ids']:
                return None
            
            # Parse metadata
            metadata = result['metadatas'][0]
            parsed_metadata = {}
            for key, value in metadata.items():
                if value.startswith('[') or value.startswith('{'):
                    try:
                        parsed_metadata[key] = json.loads(value)
                    except:
                        parsed_metadata[key] = value
                else:
                    parsed_metadata[key] = value
                
            return {
                'id': result['ids'][0],
                'description': result['documents'][0],
                'metadata': parsed_metadata
            }
            
        except Exception as e:
            print(f"Error getting example details: {str(e)}")
            traceback.print_exc()
            return None

    def perform_step_back(self, query, approved_keywords=None):
        """Perform step-back analysis using pre-approved keywords if provided"""
        try:
            if not approved_keywords:
                print("Error: No approved keywords provided")
                return None
            
            return self.step_back_analyzer.perform_analysis(query, approved_keywords)
            
        except Exception as e:
            logger.error(f"Error in step-back analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def get_examples(self, description: str, step_back_result: Dict = None, keyword_data: Dict = None, 
                   similarity_threshold: float = 0.6, return_metadata: bool = False, max_results: int = 5) -> List[Dict]:
        """
        Get relevant examples based on description, step-back analysis, and keyword data.
        
        Args:
            description: The query description
            step_back_result: Optional step-back analysis results 
            keyword_data: Optional pre-extracted keyword data
            similarity_threshold: Minimum similarity score required (default: 0.65)
            return_metadata: Whether to return extracted metadata along with examples
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            If return_metadata is False: List of relevant examples with their similarity scores
            If return_metadata is True: Tuple of (examples list, extracted metadata dict)
        """
        try:
            print("\nRetrieving similar examples...")
            print(f"Query: {description}")
            
            # Extract metadata from query
            print("\nExtracting metadata from query...")
            query_metadata = self.metadata_extractor.extract_metadata(
                description=description, 
                step_back_result=step_back_result, 
                keyword_data=keyword_data
            )
            
            # Prepare filters from step-back analysis if available
            filters = None
            if step_back_result:
                filters = {k: v for k, v in {
                    "complexity": step_back_result.get("complexity"),
                    "style": step_back_result.get("style")
                }.items() if v is not None}
            
            # Log query parameters
            self.write_debug(
                "\n=== VECTOR STORE QUERY PARAMETERS ===\n",
                f"Query Text: {description}\n",
                f"Filters: {filters}\n",
                f"Keyword Data: {keyword_data}\n",
                f"Step-back Principles: {step_back_result.get('principles', []) if step_back_result else []}\n",
                f"Similarity Threshold: {similarity_threshold}\n",
                "=" * 50 + "\n"
            )
            
            # Query the vector store
            results = self.collection.query(
                query_texts=[description],
                n_results=max_results
            )
            
            # Log raw query results
            self.write_debug(
                "\n=== VECTOR STORE QUERY RESULTS ===\n",
                f"Results count: {len(results.get('ids', [[]])[0]) if results.get('ids') else 0}\n",
                f"Result IDs: {results.get('ids', [[]])[0] if results.get('ids') else []}\n",
                f"Result Distances: {results.get('distances', [[]])[0] if results.get('distances') else []}\n",
                "=" * 50 + "\n"
            )
            
            # Check for empty results
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                print("No results found in vector store")
                return ([], query_metadata) if return_metadata else []
            
            # Prepare and filter results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                
                # Apply metadata filters
                if filters and not self._passes_filters(result['metadata'], filters):
                    continue
                    
                search_results.append(result)
            
            # Summarize raw results for logging
            print(f"\nFound {len(search_results)} raw results")
            
            # Log search results before ranking
            self.write_debug(
                "\n=== SEARCH RESULTS BEFORE RANKING ===\n",
                f"Number of results to rank: {len(search_results)}\n",
                "".join(f"Result {i}: ID={result['id']}, Distance={result['distance']}\n" 
                       for i, result in enumerate(search_results, 1)),
                "=" * 50 + "\n"
            )
            
            # Rank results
            ranked_results = self._rank_results(query_metadata, search_results)
            
            # Filter by similarity threshold
            relevant_results = [r for r in ranked_results if r['score'] >= similarity_threshold]
            
            # Log final results
            self.write_debug(
                "\n=== FINAL FILTERED RESULTS ===\n",
                f"Number of relevant results: {len(relevant_results)}\n",
                f"Similarity threshold: {similarity_threshold}\n",
                "".join(f"Result {i}: ID={r['example']['id']}, Score={r['score']:.3f}, Score breakdown: {r['score_breakdown']['component_scores']}\n" 
                       for i, r in enumerate(relevant_results, 1)),
                "=" * 50 + "\n"
            )
            
            # Display results to user
            if relevant_results:
                print(f"\nFound {len(relevant_results)} similar examples")
                for i, example in enumerate(relevant_results, 1):
                    print(f"\nExample {i} (Score: {example['score']:.3f}):")
                    print(f"ID: {example['example']['id']}")
                    print("Score Breakdown:")
                    for name, score in example['score_breakdown']['component_scores'].items():
                        print(f"  {name}: {score:.3f}")
            else:
                print("\nNo similar examples found")
            
            # Return results
            return (relevant_results, query_metadata) if return_metadata else relevant_results
            
        except Exception as e:
            print(f"Error getting examples: {str(e)}")
            traceback.print_exc()
            return ([], query_metadata) if return_metadata else []
        
    def _passes_filters(self, metadata, filters):
        """Helper method to check if an item's metadata passes the filters"""
        for key, value in filters.items():
            if key not in metadata:
                continue
            
            meta_value = metadata[key]
            # Handle potential JSON strings
            if isinstance(meta_value, str) and (meta_value.startswith('[') or meta_value.startswith('{')):
                try:
                    meta_value = json.loads(meta_value)
                except:
                    pass
            
            # Check if value matches filter
            if isinstance(meta_value, list):
                if value not in meta_value:
                    return False
            elif str(meta_value).lower() != str(value).lower():
                return False
                
        return True

    def get_all_examples(self, interactive=False) -> list:
        """Get all examples from the knowledge base.
        
        Args:
            interactive: Whether to enable interactive browsing mode
            
        Returns:
            List of dictionaries containing example data with their metadata
        """
        try:
            # Get all examples from ChromaDB
            results = self.collection.get()
            if not results or not results['ids']:
                logger.debug("No examples found in knowledge base")
                if interactive:
                    print("\nNo examples found in the knowledge base.")
                return []
            
            # Format the results
            examples = []
            for i in range(len(results['ids'])):
                try:
                    metadata = results['metadatas'][i]
                    example = {
                        'id': results['ids'][i],
                        'description': results['documents'][i],
                        'code': metadata.get('code', ''),
                        'metadata': {
                            'object_type': metadata.get('object_type', ''),
                            'features': self._parse_json_field(metadata.get('features', '[]')),
                            'timestamp': metadata.get('timestamp', ''),
                            'type': metadata.get('type', ''),
                            'user_accepted': metadata.get('user_accepted', True)
                        }
                    }
                    
                    # Add step-back analysis if available
                    step_back = metadata.get('step_back_analysis')
                    if step_back:
                        example['metadata']['step_back_analysis'] = self._parse_json_field(step_back)
                    
                    examples.append(example)
                except Exception as e:
                    logger.error(f"Error processing example {results['ids'][i]}: {str(e)}")
                    continue
            
            # Sort examples by timestamp (newest first)
            examples.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
            
            logger.debug(f"Successfully retrieved {len(examples)} examples from knowledge base")
            
            # If interactive mode is enabled, provide browsing interface
            if interactive:
                self._browse_examples(examples)
                
            return examples
            
        except Exception as e:
            logger.error(f"Error retrieving examples from knowledge base: {str(e)}")
            if interactive:
                print(f"\nError accessing knowledge base: {str(e)}")
            return []
            
    def _browse_examples(self, examples):
        """Interactive browser for knowledge base examples."""
        print("\n" + "="*50)
        print("KNOWLEDGE BASE EXPLORER")
        print("="*50)
        
        # Display examples with their descriptions and metadata
        print(f"\nFound {len(examples)} examples:")
        print("-" * 40)
        
        for i, example in enumerate(examples, 1):
            metadata = example.get('metadata', {})
            description = example.get('description', 'No description available')
            object_type = metadata.get('object_type', 'Unknown type')
            features = ', '.join(metadata.get('features', [])) or 'No features'
            timestamp = metadata.get('timestamp', 'Unknown time')
            
            print(f"\n[{i}] Example ID: {example.get('id', 'unknown')}")
            print(f"Description: {description}")
            print(f"Type: {object_type}")
            print(f"Features: {features}")
            print(f"Added: {timestamp}")
            print("-" * 40)
        
        while True:
            try:
                # Get user selection
                selection = input("\nEnter the number of the example to view (or 'q' to quit): ").lower().strip()
                
                if selection == 'q':
                    return
                
                index = int(selection) - 1
                if 0 <= index < len(examples):
                    selected = examples[index]
                    
                    # Display full example details
                    print("\n" + "="*50)
                    print("EXAMPLE DETAILS")
                    print("="*50)
                    
                    print("\nDescription:")
                    print(selected.get('description', 'No description available'))
                    
                    print("\nMetadata:")
                    metadata = selected.get('metadata', {})
                    print(f"- Object Type: {metadata.get('object_type', 'Unknown')}")
                    print(f"- Features: {', '.join(metadata.get('features', []))}")
                    print(f"- Added: {metadata.get('timestamp', 'Unknown')}")
                    print(f"- Type: {metadata.get('type', 'Unknown')}")
                    
                    if 'step_back_analysis' in metadata:
                        analysis = metadata['step_back_analysis']
                        print("\nTechnical Analysis:")
                        if 'principles' in analysis:
                            print("\nCore Principles:")
                            for p in analysis['principles']:
                                print(f"- {p}")
                        if 'abstractions' in analysis:
                            print("\nShape Components:")
                            for a in analysis['abstractions']:
                                print(f"- {a}")
                        if 'approach' in analysis:
                            print("\nImplementation Steps:")
                            for i, s in enumerate(analysis['approach'], 1):
                                print(f"{i}. {s}")
                    
                    print("\nOpenSCAD Code:")
                    print("-" * 40)
                    print(selected.get('code', 'No code available'))
                    print("-" * 40)
                    
                    # Ask if user wants to save this code to a file
                    save = input("\nWould you like to save this code to a file? (y/n): ").lower().strip()
                    if save == 'y':
                        filename = f"example_{selected.get('id', 'unknown')}.scad"
                        with open(filename, "w") as f:
                            f.write(selected.get('code', ''))
                        print(f"\nCode saved to {filename}")
                    
                    # Ask if user wants to view another example
                    again = input("\nWould you like to view another example? (y/n): ").lower().strip()
                    if again != 'y':
                        break
                else:
                    print("\nInvalid selection. Please enter a number between 1 and", len(examples))
            except ValueError:
                print("\nInvalid input. Please enter a number or 'q' to quit.")
            except Exception as e:
                print(f"\nError viewing example: {str(e)}")
                return
            
    def _parse_json_field(self, value):
        """Helper method to safely parse JSON fields"""
        if not value:
            return []
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value if value else []
            
    def input_manual_knowledge(self) -> bool:
        """Handle manual input of knowledge to add directly to the knowledge base.
        
        This method allows users to:
        1. Enter a description for a 3D model
        2. Read SCAD code from a file (add.scad)
        3. Extract and confirm keywords
        4. Perform step-back analysis
        5. Add the example to the knowledge base with metadata
        
        Returns:
            bool: True if the knowledge was successfully added, False otherwise
        """
        logger.info("Starting manual knowledge input")
        print("\nManual Knowledge Input Mode")
        print("---------------------------")
        
        # Step 1: Get description
        description = input("\nEnter the description/query for the 3D model: ").strip()
        if not description:
            logger.warning("Empty description provided")
            print("Description cannot be empty.")
            return False
            
        # Step 2: Get OpenSCAD code from add.scad
        try:
            with open("add.scad", "r") as f:
                scad_code = f.read().strip()
            if not scad_code:
                logger.warning("Empty OpenSCAD code provided")
                print("OpenSCAD code cannot be empty.")
                return False
            print("\nRead OpenSCAD code from add.scad:")
            print("-" * 52)
            print(scad_code)
            print("-" * 52 + "\n")
            
            # Validate OpenSCAD code
            logger.debug("Starting OpenSCAD code validation")
            if not self._validate_scad_code(scad_code):
                logger.warning("OpenSCAD code validation failed")
                return False
                
        except FileNotFoundError:
            logger.error("add.scad file not found")
            print("Error: add.scad file not found. Please create the file with your OpenSCAD code.")
            return False
        except Exception as e:
            logger.error(f"Error reading add.scad: {str(e)}")
            print(f"Error reading add.scad: {str(e)}")
            return False
        
        try:
            
            # Step 1: Extract keywords and get user confirmation
            print("\n" + "="*50)
            print("STEP 1: KEYWORD EXTRACTION")
            print("="*50)
            
            keyword_data = None
            max_keyword_retries = 3
            keyword_retry_count = 0
            
            # Initialize the keyword extractor if not already initialized
            from KeywordExtractor import KeywordExtractor
            if not hasattr(self, 'keyword_extractor'):
                self.keyword_extractor = KeywordExtractor()
            
            while keyword_retry_count < max_keyword_retries:
                # Extract keywords
                keyword_data = self.keyword_extractor.extract_keyword(description)
                
                print("\nKeyword Analysis Results:")
                print("-" * 30)
                print(f"Core Type: {keyword_data.get('core_type', '')}")
                print(f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}")
                print(f"Compound Type: {keyword_data.get('compound_type', '')}")
                print("-" * 30)
                
                # Ask for user confirmation
                user_input = input("\nDo you accept these keywords? (yes/no): ").lower().strip()
                
                if user_input == 'yes':
                    # Log the approved keywords
                    self.logger.log_keyword_extraction({
                        "query": {
                            "input": description,
                            "timestamp": datetime.now().isoformat()
                        },
                        "response": keyword_data,
                        "metadata": {
                            "success": True,
                            "error": None,
                            "user_approved": True
                        }
                    })
                    break
                else:
                    keyword_retry_count += 1
                    if keyword_retry_count < max_keyword_retries:
                        print("\nRetrying keyword extraction...")
                        # Ask user for refinement suggestions
                        print("Please provide any suggestions to improve the keyword extraction (or press Enter to retry):")
                        user_feedback = input().strip()
                        if user_feedback:
                            description = f"{description}\nConsider these adjustments: {user_feedback}"
                    else:
                        print("\nMaximum keyword extraction attempts reached.")
                        print("Please try again with a different description.")
                        return False

            if keyword_data is None:
                print("\nKeyword extraction failed. Please try again with a different description.")
                return False

            # Step 2: Perform step-back analysis with approved keywords
            print("\n" + "="*50)
            print("STEP 2: TECHNICAL ANALYSIS")
            print("="*50)
            
            # Initialize step-back analyzer if not already initialized
            from step_back_analyzer import StepBackAnalyzer
            from LLM import LLMProvider
            if not hasattr(self, 'step_back_analyzer'):
                llm = LLMProvider.get_llm()
                self.step_back_analyzer = StepBackAnalyzer(llm=llm, logger=self.logger)
            
            step_back_result = None
            max_step_back_retries = 3
            step_back_retry_count = 0
            
            while step_back_retry_count < max_step_back_retries:
                # Perform step-back analysis
                step_back_result = self.perform_step_back(description, keyword_data)
                if not step_back_result:
                    print("\nStep-back analysis failed. Please try again.")
                    step_back_retry_count += 1
                    continue
                
                # Ask for user confirmation
                user_input = input("\nDo you accept this technical analysis? (yes/no): ").lower().strip()
                
                if user_input == 'yes':
                    # Log the approved step-back analysis
                    self.logger.log_step_back_analysis({
                        "query": {
                            "input": description,
                            "timestamp": datetime.now().isoformat()
                        },
                        "response": {
                            "principles": step_back_result.get('principles', []),
                            "abstractions": step_back_result.get('abstractions', []),
                            "approach": step_back_result.get('approach', [])
                        },
                        "metadata": {
                            "success": True,
                            "error": None,
                            "user_approved": True
                        }
                    })
                    break
                else:
                    step_back_retry_count += 1
                    if step_back_retry_count < max_step_back_retries:
                        print("\nRetrying step-back analysis...")
                        # Ask user for refinement suggestions
                        print("Please provide any suggestions to improve the step-back analysis (or press Enter to retry):")
                        user_feedback = input().strip()
                        if user_feedback:
                            description = f"{description}\nConsider these aspects in your analysis: {user_feedback}"
                    else:
                        print("\nMaximum step-back analysis attempts reached.")
                        print("Please try again with a different description.")
                        return False

            if step_back_result is None:
                print("\nStep-back analysis failed. Please try again with a different description.")
                return False

            # Step 3: Add example to knowledge base
            print("\n" + "="*50)
            print("STEP 3: ADDING TO KNOWLEDGE BASE")
            print("="*50)
            
            # Prepare metadata for the new example
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            example_id = f"manual_{formatted_time}"
            
            example_metadata = {
                "id": example_id,
                "object_type": keyword_data.get('core_type', ''),
                "features": keyword_data.get('modifiers', []),
                "step_back_analysis": step_back_result,
                "timestamp": current_time.isoformat(),
                "user_accepted": True,
                "type": "manual"
            }
            
            # Add example to knowledge base
            success = self.add_example(description, scad_code, example_metadata)
            if success:
                print(f"\nExample successfully added to knowledge base with ID: {example_id}")
                return True
            else:
                print("\nFailed to add example to knowledge base.")
                return False
                
        except Exception as e:
            logger.error(f"Error in manual knowledge input: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            print(f"\nError saving knowledge: {str(e)}")
            return False
            
    def _validate_scad_code(self, code: str) -> bool:
        """Validate OpenSCAD code for basic syntax and structure"""
        logger.debug("Validating OpenSCAD code")
        
        if not code or len(code) < 50:  # Minimum reasonable length
            logger.warning(f"Code too short: {len(code)} characters (minimum: 50)")
            print("Error: OpenSCAD code is too short")
            return False
        
        if len(code) > 20000:  # Maximum reasonable length
            logger.warning(f"Code too long: {len(code)} characters (maximum: 20000)")
            print("Error: OpenSCAD code exceeds maximum length")
            return False
        
        required_elements = ['(', ')', '{', '}']
        missing_elements = []
        for element in required_elements:
            if element not in code:
                missing_elements.append(element)
                logger.warning(f"Missing {element} in OpenSCAD code")
                print(f"Error: OpenSCAD code is missing {element}")
        
        if missing_elements:
            return False
        
        # Check for basic OpenSCAD elements
        basic_keywords = ['module', 'function', 'cube', 'sphere', 'cylinder']
        found_keywords = [keyword for keyword in basic_keywords if keyword in code]
        
        if not found_keywords:
            logger.warning("No basic OpenSCAD elements found")
            print("Error: Code doesn't contain any basic OpenSCAD elements")
            return False
        
        logger.debug(f"Found OpenSCAD keywords: {', '.join(found_keywords)}")
        return True
        
    def delete_knowledge(self) -> bool:
        """Handle knowledge deletion with improved user experience.
        
        This method allows users to:
        1. Search examples by description
        2. Filter examples by metadata
        3. View all examples
        4. Delete selected examples
        
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        logger.info("Starting knowledge deletion")
        print("\nKnowledge Deletion Mode")
        print("----------------------")
        
        try:
            while True:
                # Show menu
                print("\nDelete Knowledge Options:")
                print("1. Search by description")
                print("2. Filter by metadata")
                print("3. View all examples")
                print("4. Return to main menu")
                
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == "4":
                    logger.info("User exited deletion mode")
                    return True
                    
                # Initialize search parameters
                search_term = None
                filters = {}
                page = 1
                page_size = 5
                
                if choice == "1":
                    search_term = input("\nEnter search term: ").strip()
                elif choice == "2":
                    print("\nAvailable filters:")
                    print("1. Style (e.g., Modern, Traditional, Minimalist)")
                    print("2. Complexity (SIMPLE, MEDIUM, COMPLEX)")
                    print("3. Object Type")
                    
                    filter_choice = input("\nEnter filter number (1-3): ").strip()
                    
                    if filter_choice == "1":
                        style = input("Enter style: ").strip()
                        filters["style"] = style
                    elif filter_choice == "2":
                        complexity = input("Enter complexity: ").strip().upper()
                        filters["complexity"] = complexity
                    elif filter_choice == "3":
                        obj_type = input("Enter object type: ").strip()
                        filters["object_type"] = obj_type
                
                # Search for examples
                while True:
                    results = self.search_examples(search_term, filters, page, page_size)
                    
                    if not results['examples']:
                        print("\nNo examples found.")
                        break
                    
                    print(f"\nShowing page {results['page']} of {results['total_pages']} (Total: {results['total']} examples)")
                    print("\nAvailable examples:")
                    print("------------------")
                    
                    for i, example in enumerate(results['examples'], 1):
                        print(f"\n{i}. ID: {example['id']}")
                        print(f"   Description: {example['description'][:100]}...")
                        print("   Metadata:")
                        print(f"   - Style: {example['metadata'].get('style', 'N/A')}")
                        print(f"   - Complexity: {example['metadata'].get('complexity', 'N/A')}")
                        print(f"   - Object Type: {example['metadata'].get('object_type', 'N/A')}")
                    
                    # Show navigation options
                    print("\nOptions:")
                    print("1-5: Select example to delete")
                    print("n: Next page")
                    print("p: Previous page")
                    print("b: Back to delete menu")
                    print("q: Return to main menu")
                    
                    action = input("\nEnter your choice: ").strip().lower()
                    
                    if action == 'q':
                        logger.info("User exited deletion mode")
                        return True
                    elif action == 'b':
                        break
                    elif action == 'n' and page < results['total_pages']:
                        page += 1
                    elif action == 'p' and page > 1:
                        page -= 1
                    elif action.isdigit():
                        index = int(action) - 1
                        if 0 <= index < len(results['examples']):
                            example = results['examples'][index]
                            
                            # Show example details
                            print("\nExample Details:")
                            print("-" * 40)
                            print(f"ID: {example['id']}")
                            print(f"Description: {example['description']}")
                            print("\nMetadata:")
                            for key, value in example['metadata'].items():
                                if key != 'code':  # Skip code for cleaner display
                                    print(f"{key}: {value}")
            
                            # Confirm deletion
                            confirm = input(f"\nAre you sure you want to delete this example? (y/n): ").strip().lower()
                            if confirm == 'y':
                                if self.delete_examples([example['id']]):
                                    print(f"\nExample {example['id']} has been deleted successfully!")
                                    logger.info(f"Deleted example {example['id']}")
                                    # Refresh the current page
                                    results = self.search_examples(search_term, filters, page, page_size)
                                    if not results['examples'] and page > 1:
                                        page -= 1
                                else:
                                    print("\nFailed to delete example.")
                                    logger.error(f"Failed to delete example {example['id']}")
                        else:
                            print("\nInvalid selection.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in delete_knowledge: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            print(f"\nError in delete_knowledge: {str(e)}")
            return False 