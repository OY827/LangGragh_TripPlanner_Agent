from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm

def create_basic_chain(provider: str = "openai", model: str = None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("human", "{question}")
    ])

    llm = get_llm(provider=provider, model=model)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain
