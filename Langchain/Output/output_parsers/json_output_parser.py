from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser



load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

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