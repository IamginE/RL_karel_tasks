import numpy as np
import torch
import os

from data_loading import load_task
from environment import Karel_Environment

def int_to_action(a:int) -> str:
    if a == 0:
        return "move"
    elif a == 1:
        return "turnLeft"
    elif a == 2:
        return "turnRight"
    elif a == 3:
        return "pickMarker"
    elif a == 4:
        return "putMarker"
    elif a == 5:
        return "finish"

def print_Karel_policy(karel_policy, path:str, max_actions:int=30) -> None:
    r"""Prints out the actions produced by a Karel policy for the tasks given by the .json file at path.
    :param karel_policy: Karel policy to be evaluated.
    :param path: Path to the .json file containing the tasks to be evaluated.
    :param max_actions: Maximum number of actions to be executed by the policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    karel_policy.to(device)
    karel_policy.eval()
    num_rows, num_cols, state = load_task(path)
    env = Karel_Environment(-1, "", num_rows, num_cols) # dummy environment
    output = '['

    for _ in range(max_actions):
            state_model = np.array([state])
            action_probs = karel_policy(torch.from_numpy(state_model).float().to(device))
            action = action_probs.argmax(dim=1, keepdim=True).flatten().item()
            output += '"' + int_to_action(action) + '", '
            state, _, terminal = env.transition(state, action)
            if terminal:
                break
    output = output[:-2] + ']'
    print(output)

def generate_solutions(karel_policy, tasks_path:str, out_path:str, min_idx:int, max_idx:int, max_actions:int=30) -> None:
    r"""Generates solutions for the tasks in the folder given by tasks_path and saves them in the folder given by out_path.
    The solutions are are saved in .json files with the same id as the corresponding task file.
    :param karel_policy: Karel policy to be evaluated.
    :param tasks_path: Path to the folder containing the tasks to be evaluated.
    :param out_path: Path to the folder where the solutions will be saved.
    :param min_idx: Minimum index of the tasks to be evaluated.
    :param max_idx: Maximum index of the tasks to be evaluated.
    :param max_actions: Maximum number of actions to be executed by the policy."""
    karel_policy.eval()
    task_path = tasks_path + "/" + str(min_idx) + "_task.json"
    num_rows, num_cols, state = load_task(task_path)
    env = Karel_Environment(-1, "", num_rows, num_cols) # dummy environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    for i in range(min_idx, max_idx):
        file = out_path + "/" + str(i) + "_seq.json"
        with open(file, 'w') as fp:
            output = '['
            task_path = tasks_path + "/" + str(i) + "_task.json"
            _, _, state = load_task(task_path)
            for _ in range(max_actions):
                state_model = np.array([state])
                action_probs = karel_policy(torch.from_numpy(state_model).float().to(device))
                action = action_probs.argmax(dim=1, keepdim=True).flatten().item()
                output += '"' + int_to_action(action) + '", '
                state, _, terminal = env.transition(state, action)
                if terminal:
                    break
            output = output[:-2] + ']'
            fp.write('{\t\n"sequence": ' + output + '\n}')
    
    
 

# from networks import Policy_Network
# actor = Policy_Network(54, 6, False)
# checkpoint_actor = torch.load("./saved_models/actor_first_100_11000.pt")
# actor.load_state_dict(checkpoint_actor['model_state_dict'])
# actor.set_softmax(True)
# 
# generate_solutions(actor, "./data/val/task", "./test", 100000, 102400)