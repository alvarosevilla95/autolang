from typing import List, Dict 
from pydantic import Field
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool 
from langchain.llms.base import BaseLLM

from .utils import parse_task_list

planning_template = """You are a task creation AI tasked with generating a full, exhaustive list of tasks to accomplish the following objective: {objective}.
The AI system that will execute these tasks will have access to the following tools:
{tool_strings}
Each task may only use a single tool, but not all tasks need to use one. The task should not specify the tool. The final task should achieve the objective. 
Each task will be performed by a capable agent, do not break the problem down into too many tasks.
Aim to keep the list short, and never generate more than 5 tasks. Your response should be each task in a separate line, one line per task.
Use the following format:
1. First task
2. Second task
"""

planning_prompt = lambda objective: PromptTemplate(
        template=planning_template,
        partial_variables={"objective": objective},
        input_variables=["tool_strings"],
        )


class PlanningChain(LLMChain):

    tool_strings: str = Field(...)

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, tools: List[Tool] , verbose: bool = True) -> "PlanningChain":
        tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
        return cls(prompt=planning_prompt(objective), llm=llm, verbose=verbose, tool_strings=tool_strings)
    
    def generate_tasks(self) -> List[Dict]:
        response = self.run(tool_strings=self.tool_strings)
        return parse_task_list(response)
