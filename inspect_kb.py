import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize ChromaDB client
persistence_dir = os.path.join(os.getcwd(), "scad_knowledge_base/chroma")
client = chromadb.PersistentClient(path=persistence_dir)

# Initialize embedding function
embedding_function = SentenceTransformerEmbeddingFunction()

# Get the collection
collection = client.get_collection(
    name="scad_examples",
    embedding_function=embedding_function
)

# Get all records
print("Fetching all records from the knowledge base...")
all_records = collection.get()

print(f"Total records found: {len(all_records['ids']) if all_records['ids'] else 0}")

# Search for propeller-related content
print("\nSearching for propeller-related content...")
propeller_records = [
    (i, doc, all_records['metadatas'][i])
    for i, doc in enumerate(all_records['documents'])
    if 'propeller' in doc.lower()
]

print(f"Found {len(propeller_records)} propeller-related documents")

# Print each propeller record
for i, (idx, doc, metadata) in enumerate(propeller_records, 1):
    print(f"\n--- Propeller Record {i} ---")
    print(f"ID: {all_records['ids'][idx]}")
    print(f"Description: {doc[:150]}...")
    
    # Print key metadata
    print("Metadata:")
    print(f"  Object Type: {metadata.get('object_type', 'N/A')}")
    print(f"  Features: {metadata.get('features', 'N/A')}")
    if 'step_back_analysis' in metadata:
        step_back = metadata['step_back_analysis']
        if step_back.startswith('{'):
            try:
                step_back_json = json.loads(step_back)
                print("  Step-back analysis:")
                if 'principles' in step_back_json:
                    print(f"    Principles: {step_back_json['principles']}")
                if 'abstractions' in step_back_json:
                    print(f"    Abstractions: {step_back_json['abstractions']}")
            except:
                print(f"  Step-back analysis: {step_back[:50]}...")
        else:
            print(f"  Step-back analysis: {step_back[:50]}...")

# Now perform query test
print("\n\nTesting query for 'propeller with 2 blades'...")
query_results = collection.query(
    query_texts=["propeller with 2 blades"],
    n_results=5
)

print(f"Query returned {len(query_results['ids'][0])} results")

# Print query results
for i, (doc_id, doc, distance) in enumerate(zip(
    query_results['ids'][0],
    query_results['documents'][0],
    query_results['distances'][0]
), 1):
    print(f"\n--- Query Result {i} ---")
    print(f"ID: {doc_id}")
    print(f"Distance: {distance}")
    print(f"Description: {doc[:150]}...")

# Compare with direct retrieval
print("\n\nTesting direct retrieval of propeller content...")
propeller_query = collection.query(
    query_texts=["propeller"],
    n_results=5
)

print(f"Direct 'propeller' query returned {len(propeller_query['ids'][0])} results")

for i, (doc_id, doc, distance) in enumerate(zip(
    propeller_query['ids'][0],
    propeller_query['documents'][0],
    propeller_query['distances'][0]
), 1):
    print(f"\n--- Direct Query Result {i} ---")
    print(f"ID: {doc_id}")
    print(f"Distance: {distance}")
    print(f"Description: {doc[:150]}...") 