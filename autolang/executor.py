from typing import List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, Tool
from langchain.llms.base import BaseLLM

from .agent.base import AutonomousAgent

class ExecutionAgent(BaseModel):

    agent: AgentExecutor = Field(...)

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, tools: List[Tool]) -> "ExecutionAgent":
        agent = AutonomousAgent.from_llm_and_tools(llm=llm, tools=tools, objective=objective)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
        return cls(agent=agent_executor)
    
    def execute_task(self, task: str, context: str) -> str:
        return self.agent.run({"input": task, "context": context})
