from collections import deque
from typing import List, Dict 
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM

from .utils import parse_task_list

reviewing_template = """Albus is a task reviewing and prioritization AI, tasked with cleaning the formatting of and reprioritizing the following tasks: {pending_tasks}.
Albus is provided with the list of completed tasks, the current pending tasks, and the information context that has been generated so far by the system.

Albus will decide if the current completed tasks and context are enough to generate a final answer. If this is the case, Albus will notify this using this exact format:
Review: Can answer

Albus will never generate the final answer.
If there is not enough information to answer, Albus will generate a new list of tasks. The tasks will be ordered by priority, with the most important task first. The tasks will be numbered, starting with {next_task_id}. The following format will be used:
Review: Must continue
#. First task
#. Second task

Albus will use the current pending tasks to generate this list, but it may remove tasks that are no longer required, or add new ones if strictly required.

The ultimate objective is: {objective}.
The following tasks have already been completed: {completed_tasks}.
This is the information context generated so far:
{context}
"""

reviewing_prompt = lambda objective: PromptTemplate(
        template=reviewing_template,
        partial_variables={"objective": objective},
        input_variables=["completed_tasks", "pending_tasks", "context", "next_task_id"],
        )

class ReviewingChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> "ReviewingChain":
        return cls(prompt=reviewing_prompt(objective), llm=llm, verbose=verbose)

    def review_tasks(self, this_task_id: int, completed_tasks: List[str], pending_tasks: List[str], context: str) -> deque[Dict]:
        next_task_id = int(this_task_id) + 1
        response = self.run(completed_tasks=completed_tasks, pending_tasks=pending_tasks, context=context, next_task_id=next_task_id)
        return deque(parse_task_list(response))
