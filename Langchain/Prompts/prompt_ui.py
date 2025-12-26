from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from prompt_generator import template
from dotenv import load_dotenv
import streamlit as st


# Load environment variables (HF API token)
load_dotenv()

# Connect to HuggingFace API endpoint with Mistral model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)
# Wrap the endpoint in ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm)


# Initialize Streamlit app
st.header ("Research Summarization using HuggingFace Chat Model")

# Define dropdowns for paper type, explanation style, and length
paper_input = st.selectbox(
    "Select Paper Type",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)


# prompt template
# template = PromptTemplate(
#     template="""Please summarize the research paper titled "{paper_input}" with the following specifications:

#     Explanation Style: {style_input}
#     Explanation Length: {length_input}

#     1  . Mathematical Details:
#     - Include relevant mathematical equations if present in the paper.
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.

# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

# Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
#     input_variables=["paper_input", "style_input", "length_input"]
# )

# invoke the prompt template with user inputs
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})



# if st.button("Get Summary"):
#     result = model.invoke(prompt)
#     st.subheader("Research Summary:")
#     st.write(result.content)


if st.button("Get Summary"):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.subheader("Research Summary:")
    st.write(result.content)