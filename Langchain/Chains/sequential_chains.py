from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate me a detailed report on {subject}",
    input_variables=["subject"],
)

prompt2 = PromptTemplate(
    template="Generate a 5 point summary of the following :\n\n{report}",
    input_variables=["report"],
)

parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"subject": "the impact of climate change on global agriculture"})

print("Final Output:\n", result)

chain.get_graph().print_ascii()