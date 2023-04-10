from collections import deque
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.agents import Tool 
from langchain.llms.base import BaseLLM
from langchain.vectorstores import VectorStore

from .executor import ExecutionAgent
from .planner import PlanningChain
from .reviewer import ReviewingChain
from .learner import LearningChain
from .printer import print_objective, print_next_task, print_task_list, print_task_result, print_end

class AutoAgent(BaseModel):

    planning_chain: PlanningChain = Field(...)
    reviewing_chain: ReviewingChain = Field(...)
    execution_agent: ExecutionAgent = Field(...)
    learning_chain: LearningChain = Field(...)

    objective: str = Field(alias="objective")
    vectorstore: Any = Field(...)

    memory: str = Field("", init=False)
    complete_list: deque = Field(default_factory=deque)
    pending_list: deque = Field(default_factory=deque)

    @classmethod
    def from_llm_and_objectives(
        cls,
        llm: BaseLLM,
        objective: str,
        tools: List[Tool],
        vectorstore: VectorStore,
        verbose: bool = False,
    ) -> "AutoAgent":
        planning_chain = PlanningChain.from_llm(llm, objective, tools=tools, verbose=verbose)
        reviewing_chain = ReviewingChain.from_llm(llm, objective, verbose=verbose)
        execution_agent = ExecutionAgent.from_llm(llm, objective, tools)
        learning_chain = LearningChain.from_llm(llm, objective, verbose=verbose)
        return cls(
            objective=objective,
            planning_chain=planning_chain,
            reviewing_chain = reviewing_chain,
            execution_agent=execution_agent,
            learning_chain=learning_chain,
            vectorstore=vectorstore,
        )

    def add_task(self, task: Dict):
        self.pending_list.append(task)

    def run(self, max_iterations: Optional[int] = None):
        num_iters = 0
        print_objective(self.objective)

        self.pending_list = deque(self.planning_chain.generate_tasks())

        while len(self.pending_list) > 0 and (max_iterations is None or num_iters < max_iterations):
            num_iters += 1
            print_task_list(self.complete_list, self.pending_list)

            task = self.pending_list.popleft()
            print_next_task(task)

            result = self.execution_agent.execute_task(self.objective, task["task_name"])
            if not result: result = "Empty result"
            print_task_result(result)

            self.complete_list.append({"task_id": task["task_id"], "task_name": task["task_name"]})
            self.memory = self.learning_chain.update_memory(
                memory=self.memory,
                observation=result,
                completed_tasks=list(self.complete_list),
                pending_tasks=[t["task_name"] for t in self.pending_list],
            )
            self.vectorstore.add_texts(
                texts=[result],
                metadatas=[{"task": task["task_name"]}],
                ids=[f"result_{task['task_id']}"],
            )
            reviewed_tasks = self.reviewing_chain.review_tasks(
                    this_task_id=len(self.complete_list) + 1,
                    completed_tasks=list(self.complete_list), 
                    pending_tasks=list(self.pending_list), 
                    context=self.memory)
            self.pending_list = deque(reviewed_tasks)

        print_end()
        final_answer = self.execution_agent.execute_task(self.objective, "Provide the final answer")
        print_task_result(final_answer)
