from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables (HF API token)
load_dotenv()


# Connect to HuggingFace API endpoint with Mistral model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
# Wrap the endpoint in ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm)

# Chat history
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    # Get user input
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    
    # Exit loop if user types 'exit'
    if user_input.lower() == 'exit':
        print("Exiting chat.")
        break
    
    # Send user input to the model and get response
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    
    # Print the model's response
    print("AI:", result.content) 