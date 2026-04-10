from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_llm(model: str = "gpt-4o-mini", **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model=model, **kwargs)


def make_chain(system_prompt: str = "You are a helpful assistant. Be concise.", model: str = "gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | get_llm(model)


def chat_loop(chain, name: str = "LangChain Chat"):
    print(f"{name} (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break
        response = chain.invoke({"input": user_input})
        print(f"AI: {response.content}\n")
