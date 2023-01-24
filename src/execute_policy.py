import numpy as np
import torch

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
    karel_policy.eval()
    num_rows, num_cols, state = load_task(path)
    env = Karel_Environment(-1, "", num_rows, num_cols) # dummy environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

from networks import Policy_Network
actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_full.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.set_softmax(True)

print_Karel_policy(actor, "./data/train/task/0_task.json")