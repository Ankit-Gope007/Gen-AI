from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Annotated, TypedDict, Optional
from Review_prompt import review_text
from dotenv import load_dotenv


load_dotenv()

# This free model doesnt work with_structured_output function , so we are just writing the code for demonstration purposes.
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_themes: Annotated[list[str], "List of key themes mentioned in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review"]
    pros: Annotated[Optional[list[str]], "List of pros mentioned in the review"]
    cons: Annotated[Optional[list[str]], "List of cons mentioned in the review"]
    name: Annotated[Optional[str], "Name of the reviewer if mentioned"]


structured_model = model.with_structured_output(Review)

review = structured_model.invoke(review_text)

print("Review :", review)