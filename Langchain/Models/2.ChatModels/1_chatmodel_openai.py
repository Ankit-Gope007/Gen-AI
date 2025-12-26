# Import ChatOpenAI for conversational model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize ChatOpenAI with GPT-4, high temperature for creativity, limited tokens
model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

# Send a prompt to generate a poem
result = model.invoke("Write a 5 line poem on cricket")

# Print the generated content
print(result.content)