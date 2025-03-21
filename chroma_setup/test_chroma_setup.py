import chromadb
from chromadb.utils import embedding_functions
import os
import uuid

class ChromaDBSetup:
    def __init__(self, persistence_dir="chroma_data"):
        self.persistence_dir = persistence_dir
        self.client = None
        self.collection = None
        self.embedding_function = None

    def setup(self):
        """Set up ChromaDB with persistence"""
        try:
            print("\n=== ChromaDB Setup and Test ===\n")
            
            # 1. Create persistence directory if it doesn't exist
            os.makedirs(self.persistence_dir, exist_ok=True)
            print(f"1. Initializing ChromaDB client with persistence directory: {self.persistence_dir}")
            
            # 2. Initialize the client with persistence
            self.client = chromadb.PersistentClient(path=self.persistence_dir)
            
            # 3. Set up the embedding function
            print("\n2. Setting up sentence transformer embedding function")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
            
            # 4. Create or get collection
            print("\n3. Creating or getting collection")
            try:
                self.collection = self.client.get_collection(
                    name="test_collection",
                    embedding_function=self.embedding_function
                )
                print("   Existing collection found!")
            except Exception as e:
                print("   Creating new collection...")
                self.collection = self.client.create_collection(
                    name="test_collection",
                    embedding_function=self.embedding_function
                )
                print("   Collection created successfully!")
            
            return True
            
        except Exception as e:
            print(f"\nError during setup: {str(e)}")
            print("\nSetup failed!")
            return False

    def test_operations(self):
        """Test basic ChromaDB operations"""
        try:
            print("\n4. Testing basic operations")
            
            # Add a test document
            test_id = str(uuid.uuid4())
            test_doc = "This is a test SCAD code: cube([10, 10, 10]);"
            
            print("\n   Adding test document...")
            self.collection.add(
                documents=[test_doc],
                metadatas=[{"type": "test"}],
                ids=[test_id]
            )
            print("   Document added successfully!")
            
            # Query the collection
            print("\n   Testing query...")
            results = self.collection.query(
                query_texts=["cube code"],
                n_results=1
            )
            print(f"   Query results: {results}")
            
            # Count documents
            print(f"\n   Total documents in collection: {self.collection.count()}")
            
            return True
            
        except Exception as e:
            print(f"\nError during operations test: {str(e)}")
            print("\nOperations test failed!")
            return False

def main():
    setup = ChromaDBSetup()
    if setup.setup():
        print("\nSetup completed successfully!")
        if setup.test_operations():
            print("\nAll tests passed successfully!")
        else:
            print("\nTests failed!")
    else:
        print("\nSetup failed!")

if __name__ == "__main__":
    main() 