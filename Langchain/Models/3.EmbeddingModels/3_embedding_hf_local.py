# Import HuggingFace embeddings for local execution
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize sentence transformer model locally
embedding = HuggingFaceEmbeddings(
    model_name ='sentence-transformers/all-MiniLM-L6-v2',
)

# List of documents to convert to embeddings
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Convert all documents into vector embeddings
vector = embedding.embed_documents(documents)

# Print the embedding vectors
print(str(vector))