# Import ChatGoogleGenerativeAI for Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize Gemini 1.5 Pro model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Send a query to the model
result = model.invoke('What is the capital of India')

# Print the model's response
print(result.content)