def print_objective(objective):
    color_print("*****Objective*****", 4)
    print(objective)

def print_task_list(complete_list, pending_list):
    color_print("*****TASK LIST*****", 5)
    print("Completed: ") 
    for task in complete_list:
        print(str(task["task_id"]) + ": " + task["task_name"])

    print("\nPending: ") 
    for task in pending_list:
        print(str(task["task_id"]) + ": " + task["task_name"])

def print_next_task(task):
    color_print("*****NEXT TASK*****", 2)
    print(str(task["task_id"]) + ": " + task["task_name"])

def print_task_result(result):
    color_print("*****TASK RESULT*****", 3)
    print(result)

def print_end(final_result):
    color_print("*****TASK ENDING*****", 1)
    print(final_result)

# leave at the end as the codes somehow screw up indenting in the rest of the file
def color_print(text: str, color: int):
    print(f"\n\033[9{color}m\033[1m{text}\033[0m\033[0m\n")
