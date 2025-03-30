import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from constant import *
from KeywordExtractor import KeywordExtractor
from conversation_logger import ConversationLogger
from datetime import datetime
from prompts import KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT, STEP_BACK_PROMPT_TEMPLATE
from llm_management import LLMProvider
from metadata_extractor import MetadataExtractor
from step_back_analyzer import StepBackAnalyzer
import re
import hashlib
from typing import Dict, List
import logging
import traceback
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class EnhancedSCADKnowledgeBase:
    def __init__(self):
        """Initialize the enhanced knowledge base with ChromaDB"""
        self.debug_log = []
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
        self.step_back_analyzer = StepBackAnalyzer(llm=self.llm, logger=self.logger)
        print("- LLM provider initialized")
        print("- Metadata extractor initialized")
        print("- Step-back analyzer initialized")
        
        # Set up prompts and extractors
        print("\nInitializing extractors...")
        self.keyword_extractor = KeywordExtractor()
        print("- Keyword extractor initialized")
        
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
            
            # Load timestamp and reset for fresh loading if needed
            self.last_processed_time = self._load_last_processed_time()
            print(f"- Last processed timestamp: {self.last_processed_time}")
            
            # Temporarily set to earlier time to force fresh load of all examples
            # This helps ensure we don't miss any examples
            original_timestamp = self.last_processed_time
            self.last_processed_time = '1970-01-01T00:00:00'
            
            # Load new examples
            print("\nChecking for new examples (with timestamp override)...")
            new_count = self._load_new_examples()
            
            # Restore timestamp if no new examples found to prevent reprocessing
            if new_count == 0:
                self.last_processed_time = original_timestamp
                self._save_last_processed_time()
                print(f"- Restored original timestamp: {self.last_processed_time}")
            
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
            # Get all user-approved examples from logs
            examples = self.logger.get_scad_generation_logs()
            if not examples:
                print("No SCAD generation logs found.")
                return 0
            
            # Debug print all entries
            print(f"Found {len(examples)} entries to process")
            for i, example in enumerate(examples, 1):
                desc = example.get('request', '')[:30] + "..." if example.get('request', '') else "[No description]"
                code_len = len(example.get('code', ''))
                timestamp = example.get('timestamp', 'Unknown')
                print(f"Entry {i}: {desc} - {code_len} chars - Timestamp: {timestamp}")
            
            new_examples = 0
            skip_stats = {
                'missing_data': 0,
                'older_timestamp': 0,
                'not_accepted': 0,
                'already_exists': 0
            }
            
            for example in examples:
                # Skip if example doesn't have required data
                timestamp = example.get('timestamp', '1970-01-01T00:00:00')
                description = example.get('request', '')
                code = example.get('code', '')
                is_accepted = example.get('user_accepted', False)
                
                # Check if we have all required data
                if not description or not code:
                    print(f"Skipping example - missing description or code")
                    skip_stats['missing_data'] += 1
                    continue
                    
                # Check if it's a newer example
                if timestamp <= self.last_processed_time:
                    print(f"Skipping example from {timestamp} - older than last processed time {self.last_processed_time}")
                    skip_stats['older_timestamp'] += 1
                    continue
                    
                # Check if it's user accepted
                if not is_accepted:
                    print(f"Skipping example - not marked as user accepted")
                    skip_stats['not_accepted'] += 1
                    continue
                
                # Generate unique ID based on content hash
                content_hash = hashlib.md5(f"{description}{code}".encode()).hexdigest()[:8]
                example_id = f"{self._generate_base_name(description)}_{content_hash}"
                
                # Check if example already exists
                if self._example_exists(example_id):
                    print(f"Skipping example with ID {example_id} - already exists in collection")
                    skip_stats['already_exists'] += 1
                    continue
                    
                # Example doesn't exist, add it
                try:
                    print(f"Adding new example with ID {example_id}")
                    metadata = {
                        "code": code,
                        "timestamp": timestamp,
                        "type": "scad_generation",
                        "user_accepted": True,
                        "description": description  # Include description in metadata for reference
                    }
                    
                    self.collection.add(
                        documents=[description],
                        metadatas=[metadata],
                        ids=[example_id]
                    )
                    new_examples += 1
                    
                    # Update last processed timestamp if this example is newer
                    if timestamp > self.last_processed_time:
                        self.last_processed_time = timestamp
                        print(f"Updating last processed time to {self.last_processed_time}")
                        self._save_last_processed_time()
                except Exception as e:
                    print(f"Error adding example {example_id}: {str(e)}")
                    continue
            
            total_skipped = sum(skip_stats.values())
            print("\nSkip Statistics:")
            print("-" * 30)
            print(f"Missing Data: {skip_stats['missing_data']}")
            print(f"Older Timestamp: {skip_stats['older_timestamp']}")
            print(f"Not User Accepted: {skip_stats['not_accepted']}")
            print(f"Already Exists: {skip_stats['already_exists']}")
            print("-" * 30)
            print(f"Summary: Processed {len(examples)} examples, Added {new_examples}, Total Skipped {total_skipped}")
            print("=" * 50)
            
            return new_examples
                
        except Exception as e:
            print(f"Error loading new examples: {str(e)}")
            traceback.print_exc()
            return 0
    
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
            
            # Update last processed timestamp
            self.last_processed = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"Error adding example: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_relevant_examples(self, query: str, similarity_threshold: float = 0.15, filters: Dict = None, keyword_data: Dict = None, step_back_result: Dict = None) -> List[Dict]:
        """
        Get relevant examples based on the query.
        
        Args:
            query: The search query
            similarity_threshold: Minimum similarity score required (default: 0.15)
            filters: Optional dictionary of metadata filters
            keyword_data: Optional pre-extracted keyword data
            
        Returns:
            List of relevant examples with their similarity scores
        """
        try:
            # Use pre-extracted keyword data if provided, otherwise extract metadata
            query_metadata = {}
            if keyword_data:
                query_metadata = {
                    'object_type': keyword_data.get('compound_type') or keyword_data.get('core_type', ''),
                    'features': keyword_data.get('modifiers', []),
                    'step_back_analysis': {
                        'core_principles': step_back_result.get('principles', []),
                        'shape_components': step_back_result.get('abstractions', []),
                        'implementation_steps': step_back_result.get('approach', [])
                    }
                }
            print(f"Query: {query}")
            print("\nExtracting metadata from query...")
            query_metadata = self.metadata_extractor.extract_metadata(description=query, step_back_result=step_back_result, keyword_data=keyword_data)
            
            # Prepare the query parameters for logging
            query_params = {
                "query_text": query,
                "n_results": 5,
                "keyword_data": keyword_data,
                "step_back_result": step_back_result,
                "filters": filters,
                "similarity_threshold": similarity_threshold
            }
            
            # Log the full query parameters
            self.write_debug(
                "\n=== VECTOR STORE QUERY PARAMETERS ===\n",
                f"Query Text: {query}\n",
                f"Filters: {filters}\n",
                f"Keyword Data: {keyword_data}\n",
                f"Step-back Principles: {step_back_result.get('principles', []) if step_back_result else []}\n",
                f"Similarity Threshold: {similarity_threshold}\n",
                "=" * 50 + "\n"
            )
            
            # Query the vector store
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Log the raw query results
            self.write_debug(
                "\n=== VECTOR STORE QUERY RESULTS ===\n",
                f"Results count: {len(results.get('ids', [[]])[0]) if results.get('ids') else 0}\n",
                f"Result IDs: {results.get('ids', [[]])[0] if results.get('ids') else []}\n",
                f"Result Distances: {results.get('distances', [[]])[0] if results.get('distances') else []}\n",
                "=" * 50 + "\n"
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
                
                # Apply metadata filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in result['metadata']:
                            meta_value = result['metadata'][key]
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
                
                search_results.append(result)
            
            print("\nRaw Results:")
            for i, result in enumerate(search_results, 1):
                print(f"\nRaw Result {i}:")
                print(f"ID: {result['id']}")
                print(f"Distance: {result['distance']}")
            
            # Log the search results before ranking
            self.write_debug(
                "\n=== SEARCH RESULTS BEFORE RANKING ===\n",
                f"Number of results to rank: {len(search_results)}\n",
                "".join(f"Result {i}: ID={result['id']}, Distance={result['distance']}\n" 
                       for i, result in enumerate(search_results, 1)),
                "=" * 50 + "\n"
            )
            
            # Rank results
            ranked_results = self._rank_results(query_metadata, search_results)
            
            """
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
            """
            
            # Log the ranked results before filtering
            self.write_debug(
                "\n=== RANKED RESULTS BEFORE FILTERING ===\n",
                f"Number of ranked results: {len(ranked_results)}\n",
                "".join(f"Rank {i}: ID={result['example']['id']}, Score={result['score']:.3f}\n" 
                       for i, result in enumerate(ranked_results, 1)),
                "=" * 50 + "\n"
            )
            
            # Filter by similarity threshold
            relevant_results = [
                result for result in ranked_results
                if result['score'] >= similarity_threshold
            ]
            
            # Log the final filtered results
            self.write_debug(
                "\n=== FINAL FILTERED RESULTS ===\n",
                f"Number of relevant results: {len(relevant_results)}\n",
                f"Similarity threshold: {similarity_threshold}\n",
                "".join(f"Result {i}: ID={result['example']['id']}, Score={result['score']:.3f}, Score breakdown: {result['score_breakdown']['component_scores']}\n" 
                       for i, result in enumerate(relevant_results, 1)),
                "=" * 50 + "\n"
            )
            
            print(f"\nFound {len(relevant_results)} relevant examples (threshold: {similarity_threshold}):")
            for result in relevant_results:
                print(f"\nExample (Score: {result['score']:.3f}):")
                print(f"ID: {result['example']['id']}")
                print("Score Breakdown:")
                for name, score in result['score_breakdown']['component_scores'].items():
                    print(f"  {name}: {score:.3f}")
                
            return relevant_results
            
        except Exception as e:
            print(f"Error getting relevant examples: {str(e)}")
            traceback.print_exc()
            return []
            
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
                
                query_metadata.setdefault('features', [])
                query_metadata.setdefault('geometric_properties', [])
                query_metadata.setdefault('step_back_analysis', {})
                query_metadata.setdefault('style', 'Modern')
                query_metadata.setdefault('complexity', 'SIMPLE')

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
                except:
                    result_metadata['step_back_analysis'] = {
                        'core_principles': [],
                        'shape_components': [],
                        'implementation_steps': []
                    }

                # Ensure step-back analysis structure
                for metadata in [result_metadata, query_metadata]:
                    metadata['step_back_analysis'].setdefault('core_principles', [])
                    metadata['step_back_analysis'].setdefault('shape_components', [])
                    metadata['step_back_analysis'].setdefault('implementation_steps', [])

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

                # Calculate final score with weights
                weights = {
                    'component_match': 0.35,
                    'geometric_match': 0.25,
                    'step_back_match': 0.20,
                    'feature_match': 0.15,
                    'style_match': 0.03,
                    'complexity_match': 0.02
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

    def get_similar_examples(self, description: str, step_back_result: Dict = None, keyword_data: Dict = None, similarity_threshold: float = 0.7, return_metadata: bool = False) -> List[Dict]:
        """
        Get similar examples based on description, step-back analysis, and keyword data.
        
        Args:
            description: The query description
            step_back_result: Optional step-back analysis results containing complexity and style
            keyword_data: Optional pre-extracted keyword data
            similarity_threshold: Minimum similarity score required (default: 0.7)
            return_metadata: Whether to return extracted metadata along with examples
            
        Returns:
            If return_metadata is False: List of relevant examples with their similarity scores
            If return_metadata is True: Tuple of (examples list, extracted metadata dict)
        """
        try:
            print("\nRetrieving similar examples...")
            
            # Prepare filters from step-back analysis if available
            filters = None
            if step_back_result:
                filters = {
                    "complexity": step_back_result.get("complexity", None),
                    "style": step_back_result.get("style", None)
                }
                # Remove None values
                filters = {k: v for k, v in filters.items() if v is not None}
            
            # Get relevant examples
            examples = self.get_relevant_examples(
                description,
                filters=filters,
                keyword_data=keyword_data,
                similarity_threshold=similarity_threshold,
                step_back_result=step_back_result
            )
            
            # Get the extracted metadata from the last query
            query_metadata = {}
            if keyword_data:
                query_metadata = {
                    'object_type': keyword_data.get('compound_type') or keyword_data.get('core_type', ''),
                    'features': keyword_data.get('modifiers', []),
                    'step_back_analysis': {
                        'core_principles': step_back_result.get('principles', []),
                        'shape_components': step_back_result.get('abstractions', []),
                        'implementation_steps': step_back_result.get('approach', [])
                    }
                }
            
            if examples:
                print(f"\nFound {len(examples)} similar examples")
                for i, example in enumerate(examples, 1):
                    print(f"\nExample {i} (Score: {example['score']:.3f}):")
                    print(f"ID: {example['example']['id']}")
                    print("Score Breakdown:")
                    for name, score in example['score_breakdown']['component_scores'].items():
                        print(f"  {name}: {score:.3f}")
            else:
                print("\nNo similar examples found")
            
            if return_metadata:
                return examples, query_metadata
            return examples
            
        except Exception as e:
            print(f"Error getting similar examples: {str(e)}")
            traceback.print_exc()
            if return_metadata:
                return [], {}
            return []

    def prepare_generation_inputs(self, description: str, examples: List[Dict], step_back_result: Dict = None) -> Dict:
        """
        Prepare inputs for the code generation prompt.
        
        Args:
            description: The original query/description
            examples: List of similar examples found
            step_back_result: Optional step-back analysis results
            
        Returns:
            Dictionary containing all inputs needed for the generation prompt
        """
        try:
            # Format step-back analysis if available
            step_back_text = ""
            if step_back_result:
                principles = step_back_result.get('principles', [])
                abstractions = step_back_result.get('abstractions', [])
                approach = step_back_result.get('approach', [])
                
                step_back_text = f"""
                CORE PRINCIPLES:
                {chr(10).join(f'- {p}' for p in principles)}
                
                SHAPE COMPONENTS:
                {chr(10).join(f'- {a}' for a in abstractions)}
                
                IMPLEMENTATION STEPS:
                {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(approach))}
                """
            
            # Format examples for logging
            examples_text = []
            for ex in examples:
                example_id = ex.get('example', {}).get('id', 'unknown')
                score = ex.get('score', 0.0)
                score_breakdown = ex.get('score_breakdown', {})
                
                example_text = f"""
                Example ID: {example_id}
                Score: {score:.3f}
                Component Scores:
                {chr(10).join(f'  - {name}: {score:.3f}' for name, score in score_breakdown.get('component_scores', {}).items())}
                """
                examples_text.append(example_text)
            
            # Prepare the complete inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples,
                "request": description,
                "step_back_analysis": step_back_text.strip() if step_back_text else ""
            }
            
            # Log the complete analysis and examples
            """
            print("\nStep-back Analysis:")
            print(step_back_text.strip() if step_back_text else "No step-back analysis available")
            """
            
            print("\nRetrieved Examples:")
            if examples_text:
                print("\n".join(examples_text))
            else:
                print("No examples found")
            
            return inputs
            
        except Exception as e:
            print(f"Error preparing generation inputs: {str(e)}")
            traceback.print_exc()
            return {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": [],
                "request": description,
                "step_back_analysis": ""
            } 

    def get_all_examples(self) -> list:
        """Get all examples from the knowledge base.
        
        Returns:
            List of dictionaries containing example data with their metadata
        """
        try:
            # Get all examples from ChromaDB
            results = self.collection.get()
            if not results or not results['ids']:
                logger.debug("No examples found in knowledge base")
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
            return examples
            
        except Exception as e:
            logger.error(f"Error retrieving examples from knowledge base: {str(e)}")
            return []
    
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