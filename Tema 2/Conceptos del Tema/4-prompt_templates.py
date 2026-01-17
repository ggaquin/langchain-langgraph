from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system","Eres un traductor de español a inglés muy preciso"),
    ("human","{texto}")
])

messages = chat_prompt.format_messages(texto="Hola mundo, ¿como estan?")

for m in messages:
    print(f"{type(m)}: {m.content}")