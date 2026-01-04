from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field



load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Character(BaseModel):
    name: str
    age: int = Field( gt=0 , lt=150 , description="Age must be between 0 and 150")
    city: str

parser = PydanticOutputParser(pydantic_object=Character)

template = PromptTemplate(
    template="""give me a name , age and city of a fictional character \n {format_instructions}""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_parsed_output = parser.parse(result.content)

chain = template | model | parser

final_parsed_output = chain.invoke({})

print("Final Parsed Output:\n", final_parsed_output)