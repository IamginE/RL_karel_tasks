from data_loading import load_seq, load_task
from environment import Karel_Environment

import numpy as np
import torch
from typing import Tuple

def eval_policy(policy, filepath:str, start:int, end:int, max_actions:int, num_rows:int, 
    num_cols:int, gamma:float, device) -> Tuple[float, int, int]:
    r"""Evaluates a policy on a given set of tasks.
    :param policy: Policy to evaluate.
    :param filepath: Path to the data.
    :param start: Start index of the tasks to evaluate on.
    :param end: End index of the tasks to evaluate on.
    :param max_actions: Maximum number of actions per episode to perform, i.e., cut-off to prevent infinite runs.
    :param num_rows: Number of rows in the grid.
    :param num_cols: Number of columns in the grid.
    :param gamma: Discount factor.
    :param device: Device used in training (cpu or gpu).
    :return: Average return, number of shortest solutions, number of solved tasks.
    """
    policy.eval()
    env = Karel_Environment(-1, "", 4, 4) # we don't need to sample initial states here
    num_solved = 0
    num_shortest = 0
    returns = 0
    size = num_rows * num_cols
    num = end - start
    for idx in range(start, end, 1):
        seq_path = filepath + "/seq/" + str(idx) + "_seq.json"
        task_path = filepath + "/task/" + str(idx) + "_task.json"
        _, _, state =  load_task(task_path)
        seq = load_seq(seq_path)
        shortest_seq_len = len(seq)
        terminal = False
        seq_len = 0
        episode_reward = 0
        for i in range(max_actions):
            state_model = np.array([state])
            action_probs = policy(torch.from_numpy(state_model).float().to(device))
            action = action_probs.argmax(dim=1, keepdim=True).flatten().item()
            state, reward, terminal = env.transition(state, action)
            episode_reward += reward * gamma**i
            seq_len += 1
            if terminal:
                break
        if action == 5 and np.array_equal(state[size:2*size], state[2*size:3*size]) and np.array_equal(state[3*size:3*size+3], state[3*size+3:3*size+6]):
            num_solved += 1
            if seq_len == shortest_seq_len:
                num_shortest += 1
        returns += episode_reward
    print('Avg. Return: {:.6f} \tShortest Sequences: {} \tNumber solved: {}/{}'.format(
        returns/num, num_shortest, num_solved, num
    ))
    return returns/num, num_shortest, num_solved






