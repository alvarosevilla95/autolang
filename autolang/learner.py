from typing import List 
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM

learning_template = """Cass is an AI specialized in information consolidation, part of a larger system that is solving a complex problem in multiple steps. Cass is provided the current information context, and the result of the latest step, and updates the context incorporating the result. It is also provided the list of completed and still pending tasks. This is for Cass to better decide what is the relevant information that should be carried over, but does not include the tasks themselves, as they are provided separately to the executor."
The ultimate objective is: {objective}.
Completed tasks: {completed_tasks}
The last task output was:
{last_output}

The list of pending tasks: {pending_tasks}
Current context to update:
{context}

Cass will generate an updated context. This context will replace the current context.
Cass: """

learning_prompt = lambda objective: PromptTemplate(
        template=learning_template,
        partial_variables={"objective": objective},
        input_variables=["completed_tasks", "pending_tasks", "last_output", "context"],
        )

class LearningChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> "LearningChain":
        return cls(prompt=learning_prompt(objective), llm=llm, verbose=verbose)
    
    def update_memory(self, memory: str, observation: str, completed_tasks: List[str], pending_tasks: List[str]):
        return self.run(
                completed_tasks=completed_tasks, 
                pending_tasks=pending_tasks, 
                last_output=observation, 
                context=memory
                )


