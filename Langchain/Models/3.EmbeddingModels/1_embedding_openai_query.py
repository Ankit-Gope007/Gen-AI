# Import OpenAI embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize embedding model with 32 dimensions
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# Convert a single query text into vector embedding
result = embedding.embed_query("Delhi is the capital of India")

# Print the embedding vector
print(str(result))