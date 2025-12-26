# # Import required libraries for embeddings and similarity calculation
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load environment variables (API keys)
# load_dotenv()

# # Initialize embedding model with 300 dimensions for better accuracy
# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# # Collection of cricket player descriptions
# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# # Query to find similar document
# query = 'tell me about bumrah'

# # Convert all documents and query to embeddings
# doc_embeddings = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# # Calculate cosine similarity between query and each document
# scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# # Find the document with highest similarity score
# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

# # Display query, most similar document, and similarity score
# print(query)
# print(documents[index])
# print("similarity score is:", score)



from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about sachin'

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)