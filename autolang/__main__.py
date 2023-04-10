import faiss
import readline # for better CLI experience
from typing import List
from langchain import FAISS, InMemoryDocstore
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.base import BaseLLM 

from .auto import AutoAgent

objective = input('What is my purpose? ')

llm: BaseLLM = ChatOpenAI(model_name="gpt-4", temperature=0, request_timeout=120) # type: ignore 
embeddings = OpenAIEmbeddings() # type: ignore

tools: List[Tool] = load_tools(["python_repl", "human"], llm=llm)  # type: ignore

index = faiss.IndexFlatL2(1536)
docstore = InMemoryDocstore({})
vectorstore = FAISS(embeddings.embed_query, index, docstore, {}) 

agent = AutoAgent.from_llm_and_objectives(llm, objective, tools, vectorstore, verbose=True) 

agent.run()
