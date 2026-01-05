from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
# llm2 = HuggingFaceEndpoint(
#     repo_id="MiniMaxAI/MiniMax-M2.1",
#     task="text-generation"
# )

model1 = ChatHuggingFace(llm=llm1)
# model2 = ChatHuggingFace(llm=llm2)

# either positive or negative
class reviewClassification(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field( description="Sentiment should be either Positive or Negative")

parser = PydanticOutputParser(pydantic_object=reviewClassification)

prompt1 = PromptTemplate(
    template="Classify the following sentiment as Positive or Negative: {review}\n\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


parser2 = StrOutputParser()




classifier_chain = prompt1 | model1 | parser


prompt2 = PromptTemplate(
    template="Write an appropriate positive review about {review}.",
    input_variables=["review"],
)
prompt3 = PromptTemplate(
    template="Write an appropriate negative review about {review}.",
    input_variables=["review"],
)


branch_chain = RunnableBranch(
    (lambda x :x.sentiment == "Positive", prompt2 | model1 | parser2 ),
    (lambda x :x.sentiment == "Negative", prompt3 | model1 | parser2 ),
    RunnableLambda(lambda x: "No valid sentiment detected." )
)

chain = classifier_chain | branch_chain

final_output = chain.invoke({"review": "The product was not good , had terrible quality and broke within a week."})
print("Final Output:\n", final_output)

chain.get_graph().print_ascii()