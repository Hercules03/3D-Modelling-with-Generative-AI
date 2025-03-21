# ChromaDB Setup with Persistent Storage

This directory contains the setup for ChromaDB with persistent storage, which will be used for storing and retrieving SCAD examples using vector embeddings.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the test script:
```bash
python test_chroma_setup.py
```

## Directory Structure

- `requirements.txt`: Contains all necessary Python dependencies
- `test_chroma_setup.py`: Test script to verify ChromaDB setup
- `chroma_data/`: Directory where ChromaDB will store its persistent data (created automatically)

## What the Test Does

1. Initializes ChromaDB with persistent storage
2. Sets up sentence transformer embeddings
3. Creates a test collection
4. Tests basic operations:
   - Adding a document
   - Querying the collection
   - Checking document count

## Expected Output

If everything works correctly, you should see output similar to:
```
=== ChromaDB Setup and Test ===

1. Initializing ChromaDB client with persistence directory: chroma_data

2. Setting up sentence transformer embedding function

3. Creating or getting collection
   - Created new collection: test_collection

4. Testing basic operations
   - Adding test document
   - Testing query

   Query Results:
   - Found document: A simple test cup
   - Distance: [distance value]
   - Metadata: {...}

   - Total documents in collection: 1

=== Setup and tests completed successfully! ===
```

## Troubleshooting

1. If you get an error about SQLite version, make sure you have SQLite 3.35 or higher
2. If you get memory errors, ensure you have enough RAM available
3. If the persistent directory isn't created, check your write permissions

## Next Steps

After confirming the setup works:
1. Integrate ChromaDB into the main SCAD knowledge base
2. Implement semantic search for SCAD examples
3. Add metadata filtering and advanced querying 