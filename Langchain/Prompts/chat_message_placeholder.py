from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{user_input}"),
])

chat_history = []

# load chat history  from chat history.txt
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

# Making the prompt with chat history
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "user_input": "Still persists."
})

print(prompt)
