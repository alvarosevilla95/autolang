import os
import faiss
import readline # for better CLI experience
from typing import List
from langchain import FAISS, InMemoryDocstore
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.base import BaseLLM 

from .auto import AutoAgent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", default="gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

objective = input('What is my purpose? ')


llm: BaseLLM = ChatOpenAI(model_name=OPENAI_API_MODEL, temperature=0, request_timeout=120) # type: ignore 
embeddings = OpenAIEmbeddings() # type: ignore

"""
Customize the tools the agent uses here. Here are some others you can add:

os.environ["WOLFRAM_ALPHA_APPID"] = "<APPID>"
os.environ["SERPER_API_KEY"] = "<KEY>"

tool_names = ["terminal", "requests", "python_repl", "human", "google-serper", "wolfram-alpha"]
"""

tool_names = ["python_repl", "human"]

tools: List[Tool] = load_tools(tool_names, llm=llm)  # type: ignore

index = faiss.IndexFlatL2(1536)
docstore = InMemoryDocstore({})
vectorstore = FAISS(embeddings.embed_query, index, docstore, {}) 

agent = AutoAgent.from_llm_and_objectives(llm, objective, tools, vectorstore, verbose=True) 

agent.run()
