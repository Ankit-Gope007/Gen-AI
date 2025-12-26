# Import HuggingFace components for API-based chat
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables (HF API token)
load_dotenv()

# Connect to HuggingFace API endpoint with TinyLlama model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

# Wrap the endpoint in ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm)

# Send a query and get response
result = model.invoke("What is the capital of India")

# Print the model's response
print(result.content)