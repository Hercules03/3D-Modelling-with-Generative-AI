import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from constant import *
from KeywordExtractor import KeywordExtractor
from ExampleValidator import ExampleValidator
from conversation_logger import ConversationLogger
from datetime import datetime
from prompts import VALIDATION_PROMPT, KEYWORD_EXTRACTOR_PROMPT
from LLM import LLMProvider

class EnhancedSCADKnowledgeBase:
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
        
        # Initialize LLM
        print("\nInitializing LLM provider...")
        self.llm = LLMProvider.get_llm()
        print("- LLM provider initialized")
        
        # Set up prompts
        print("\nInitializing prompts...")
        self.validation_prompt = VALIDATION_PROMPT
        self.keyword_prompt = KEYWORD_EXTRACTOR_PROMPT
        print("- Validation prompt loaded")
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
            
            # Close the client (no need to call persist() as PersistentClient handles this automatically)
            print("- Closing ChromaDB client...")
            if hasattr(self, 'client'):
                self.client.reset()
                self.client = None
                self.collection = None
            
            print("=== Cleanup Complete ===")
            
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
    
    def add_example(self, description, code):
        """Add a new example to the knowledge base"""
        try:
            # Generate unique ID based on content hash
            import hashlib
            content_hash = hashlib.md5(f"{description}{code}".encode()).hexdigest()[:8]
            example_id = f"{self._generate_base_name(description)}_{content_hash}"
            
            # Add to ChromaDB
            timestamp = datetime.now().isoformat()
            self.collection.add(
                documents=[description],
                metadatas=[{
                    "code": code,
                    "timestamp": timestamp,
                    "type": "scad_generation",
                    "user_accepted": True
                }],
                ids=[example_id]
            )
            
            # Update last processed timestamp
            self.last_processed_time = timestamp
            self._save_last_processed_time()
            
            print(f"\nExample added successfully with ID: {example_id}")
            return True
            
        except Exception as e:
            print(f"\nError adding example: {e}")
            return False
    
    def get_relevant_examples(self, query, max_examples=2):
        """Get relevant examples for the given query"""
        try:
            print(f"\nSearching for examples matching: '{query}'")
            
            # Get candidates from ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=5  # Get more candidates for validation
            )
            
            if not results or not results['documents']:
                print("No examples found in knowledge base.")
                return ""
            
            validated_examples = []
            
            # Validate each example
            for i, (desc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                # Clean up description - remove any prompt templates or system messages
                clean_desc = desc
                if "Human:" in clean_desc:
                    clean_desc = clean_desc.split("Human:")[-1].strip()
                if "Project Requirements:" in clean_desc:
                    clean_desc = clean_desc.split("Project Requirements:")[-1].strip()
                if "\n" in clean_desc:
                    clean_desc = clean_desc.split("\n")[0].strip()
                
                code = metadata.get('code', '')
                if not code:
                    continue
                
                # Get validation result using LLM
                try:
                    validation_prompt = self.validation_prompt.format(
                        query=query,
                        description=clean_desc,
                        object_type=self._extract_object_type(clean_desc)
                    )
                    
                    response = self.llm.invoke(validation_prompt)
                    result = response.content.strip().lower()
                    
                    # Log validation decision quietly
                    self.logger.log_validation({
                        "desired_object": query,
                        "example_object": clean_desc,
                        "decision": "USEFUL" if result == 'useful' else "UNUSEFUL",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if result == 'useful':
                        print(f"✓ Found relevant example: '{clean_desc}'")
                        validated_examples.append(f"Example {i+1}:\n{code}\n")
                        if len(validated_examples) >= max_examples:
                            break
                    
                except Exception:
                    continue  # Skip this example if validation fails
            
            if validated_examples:
                print(f"\nUsing {len(validated_examples)} relevant examples")
                return "\n".join(validated_examples)
            else:
                print("\nNo relevant examples found")
                return ""
                
        except Exception as e:
            print(f"Error during example search: {str(e)}")
            return ""
            
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