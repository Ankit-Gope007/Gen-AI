from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
# Not working with StructuredOutputParser IDK why some issues with import maybe

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description= "Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description= "Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description= "Fact 3 about the topic"),
    ResponseSchema(name="fact_4", description= "Fact 4 about the topic"),
    ResponseSchema(name="fact_5", description= "Fact 5 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema) 


template = PromptTemplate(
    template="""give me a 5 facts about {topic}\n {format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_parsed_output = parser.parse(result.content)

chain = template | model | parser

final_parsed_output = chain.invoke({"topic": "space exploration"})

print("Final Parsed Output:\n", final_parsed_output) 