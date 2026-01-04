from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from typing import Annotated, TypedDict, Optional
from pydantic import BaseModel, Field
from Review_prompt import review_text
from dotenv import load_dotenv


load_dotenv()

# This free model doesnt work with_structured_output function , so we are just writing the code for demonstration purposes.
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes: list[str] = Field(description="List of key themes mentioned in the review")
    summary: str = Field( description="A brief summary of the review")
    sentiment: str = Field(description="The sentiment of the review")
    pros: list[str] | None = Field(None, description="List of pros mentioned in the review")
    cons: list[str] | None = Field(None, description="List of cons mentioned in the review")
    name: str | None = Field(None, description="Name of the reviewer if mentioned")
    


structured_model = model.with_structured_output(Review)

review = structured_model.invoke(review_text)

print("Review :", review)