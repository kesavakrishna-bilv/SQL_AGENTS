from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Load schema and documentation content
with open('database_documentation.md', 'r') as f:
    documentation_text = f.read()
with open('database_schema.json', 'r') as f:
    schema_text = json.dumps(json.load(f), indent=4)

# Combine schema and documentation into one list of "documents"
documents = [schema_text, documentation_text]

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata for later retrieval
faiss.write_index(index, 'database_faiss.index')
with open('documents.json', 'w') as f:
    json.dump(documents, f)
