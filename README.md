# Autolang

Another take on BabyAGI, focused on workflows that complete. Powered by langchain. 

Here's a simple demo: https://twitter.com/pictobit/status/1645504308874563584

## Running

* (Optional): Customize the tools provided in [\_\_main\_\_.py](autolang/\_\_main\_\_.py)
* Install dependencies and run
```
pip install -r requirements.txt
python -m autolang
```
Or run with docker:
```
./run_docker.sh
```
## Architecture

<p align="center">
    <img src="https://github.com/alvarosevilla95/autolang/blob/master/assets/diagram.svg">
</p>

### Planner
Runs once at the start, it thinks of a strategy to solve the problem, and produces a task list.

### Executor
A custom langchain agent, which implements ReAct to solve a single task in the plan. It can be provided any tools in the langchain format.

### Learner
Here's the interesting part. The system holds an information context string, which starts empty. 
After each step, the learner merges the result with the current context, as a sort of medium-term memory

### Reviewer
Assesses the current task list, based on the current completed tasks and generated info context, and reprioritizes the pending tasks accordingly

## Next steps
Right now, the main limitation is the limited info context. As a next step, I'm planning on adding a "long term memory agent", that extracts information from the context, replacing it with a key. The executor agent will be provided a tool to retrieve these saved snippets if required.


