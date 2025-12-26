# Import HuggingFace components for local model execution
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set custom cache directory for downloaded models
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Load model locally with generation parameters
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,  # Controls randomness
        max_new_tokens=100  # Limits response length
    )
)
# Wrap the pipeline in ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm)

# Send a query and get response
result = model.invoke("What is the capital of India")

# Print the model's response
print(result.content)