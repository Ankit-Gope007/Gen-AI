# Import OpenAI embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize embedding model with 32 dimensions
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# List of documents to convert to embeddings
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Convert all documents into vector embeddings
result = embedding.embed_documents(documents)

# Print the embedding vectors for all documents
print(str(result))