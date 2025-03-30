import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from metadata_extractor import MetadataExtractor
from KeywordExtractor import KeywordExtractor
from LLM import LLMProvider
from conversation_logger import ConversationLogger
from LLMPromptLogger import LLMPromptLogger
import traceback
from fuzzywuzzy import fuzz

# Initialize components
conversation_logger = ConversationLogger()
prompt_logger = LLMPromptLogger() 
keyword_extractor = KeywordExtractor()
metadata_extractor = MetadataExtractor("anthropic", conversation_logger, prompt_logger)

# Initialize ChromaDB client
persistence_dir = os.path.join(os.getcwd(), "scad_knowledge_base/chroma")
client = chromadb.PersistentClient(path=persistence_dir)
embedding_function = SentenceTransformerEmbeddingFunction()
collection = client.get_collection(name="scad_examples", embedding_function=embedding_function)

# Get the propeller example
propeller_record = None
all_records = collection.get()
for i, doc in enumerate(all_records['documents']):
    if 'propeller' in doc.lower():
        propeller_record = {
            'id': all_records['ids'][i],
            'document': doc,
            'metadata': all_records['metadatas'][i]
        }
        break

# Test query as in the knowledge base class
query_description = "propeller with 2 blades"
query_metadata = metadata_extractor.extract_metadata(description=query_description)

print(f"Query metadata for '{query_description}':")
print(json.dumps(query_metadata, indent=2))

if propeller_record:
    print(f"\nPropeller record metadata:")
    print(json.dumps(propeller_record['metadata'], indent=2))

    # Perform similarity calculations
    def calculate_text_similarity(list1, list2):
        if not list1 or not list2:
            return 0.0
            
        # Convert to lists if strings
        if isinstance(list1, str):
            list1 = [list1]
        if isinstance(list2, str):
            list2 = [list2]
            
        # Calculate max similarity between any pair of items
        max_similarity = 0.0
        for item1 in list1:
            for item2 in list2:
                similarity = fuzz.ratio(str(item1).lower(), str(item2).lower()) / 100.0
                max_similarity = max(max_similarity, similarity)
                
        return max_similarity

    # Calculate component match score
    result_components = []
    if 'step_back_analysis' in propeller_record['metadata']:
        step_back = propeller_record['metadata']['step_back_analysis']
        if isinstance(step_back, str):
            try:
                step_back_json = json.loads(step_back)
                if 'shape_components' in step_back_json:
                    result_components = step_back_json['shape_components']
            except:
                pass
    
    query_components = query_metadata.get('step_back_analysis', {}).get('shape_components', [])
    component_match = calculate_text_similarity(query_components, result_components)
    
    # Calculate feature match
    result_features = propeller_record['metadata'].get('features', [])
    if isinstance(result_features, str):
        try:
            result_features = json.loads(result_features)
        except:
            result_features = [result_features]
    query_features = query_metadata.get('features', [])
    feature_score = calculate_text_similarity(query_features, result_features)

    # Calculate object type match
    query_object_type = query_metadata.get('object_type', '')
    result_object_type = propeller_record['metadata'].get('object_type', '')
    object_type_score = fuzz.ratio(query_object_type.lower(), result_object_type.lower()) / 100.0

    print(f"\nSimilarity Analysis:")
    print(f"  Component match: {component_match:.3f}")
    print(f"  Feature match: {feature_score:.3f}")
    print(f"  Object Type match: {object_type_score:.3f}")

    # Calculate final score with standard weights
    weights = {
        'component_match': 0.35,
        'step_back_match': 0.20,
        'geometric_match': 0.25,
        'feature_match': 0.15,
        'style_match': 0.03,
        'complexity_match': 0.02
    }

    final_score = (
        weights['component_match'] * component_match +
        weights['feature_match'] * feature_score 
    )

    print(f"\nEstimated final score (partial): {final_score:.3f}")

    # Check raw distance from vector DB
    try:
        raw_distance = collection.query(
            query_texts=[query_description],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where={"id": propeller_record['id']}
        )
        
        if raw_distance['ids'][0]:
            print(f"\nRaw vector distance: {raw_distance['distances'][0][0]}")
    except Exception as e:
        print(f"Error getting raw distance: {str(e)}")
else:
    print("No propeller record found.") 