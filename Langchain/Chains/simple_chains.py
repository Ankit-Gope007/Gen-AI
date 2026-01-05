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


prompt = PromptTemplate(
    template="Tell me 5 interesting facts about {subject}",
    input_variables=["subject"],
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"subject": "the Eiffel Tower"})

print("Final Output:\n", result)

chain.get_graph().print_ascii()