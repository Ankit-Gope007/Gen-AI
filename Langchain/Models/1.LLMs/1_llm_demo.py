# Import OpenAI LLM wrapper from langchain
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize OpenAI LLM with GPT-3.5 turbo instruct model
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# Invoke the LLM with a question and get the response
result = llm.invoke("What is the capital of India")

# Display the result
print(result)