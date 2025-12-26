# Import ChatAnthropic for Claude model
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize Claude 3.5 Sonnet model
model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Send a query to the model
result = model.invoke('What is the capital of India')

# Print the model's response
print(result.content)