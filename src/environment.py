from data_loading import load_seq, load_task

import numpy as np
from typing import Tuple
import random

def state_to_string(state:np.ndarray, include_post:bool, num_rows:int, num_cols:int) -> str:
    state_str = "Current state:\n"
    marker_str = "x"
    wall_str = "#"
    default_str = "."
    size = num_rows * num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            pos = row*num_cols + col
            if state[pos] == 1:
                state_str += wall_str
            elif row == state[3*size] and col == state[3*size+1]:
                if state[size + pos] == 1:     
                    if state[3*size+2] == 0:
                        state_str += "N"
                    elif state[3*size+2] == 1:
                        state_str += "E"
                    elif state[3*size+2] == 2:
                        state_str += "S"
                    elif state[3*size+2] == 3:
                        state_str += "W"
                else:  
                    if state[3*size+2] == 0:
                        state_str += "n"
                    elif state[3*size+2] == 1:
                        state_str += "e"
                    elif state[3*size+2] == 2:
                        state_str += "s"
                    elif state[3*size+2] == 3:
                        state_str += "w"
            elif state[size + pos] == 1:
                state_str += marker_str
            else:
                state_str += default_str
        state_str += "\n"
    if include_post:
        state_str += "Post-grid:\n"
        for row in range(num_rows):
            for col in range(num_cols):
                pos = row*num_cols + col
                if state[pos] == 1:
                    state_str += wall_str
                elif row == state[3*size+3] and col == state[3*size+4]:
                    if state[2*size + pos] == 1:     
                        if state[3*size+5] == 0:
                            state_str += "N"
                        elif state[3*size+5] == 1:
                            state_str += "E"
                        elif state[3*size+5] == 2:
                            state_str += "S"
                        elif state[3*size+5] == 3:
                            state_str += "W"
                    else:  
                        if state[3*size+5] == 0:
                            state_str += "n"
                        elif state[3*size+5] == 1:
                            state_str += "e"
                        elif state[3*size+5] == 2:
                            state_str += "s"
                        elif state[3*size+5] == 3:
                            state_str += "w"
                elif state[2*size + pos] == 1:
                    state_str += marker_str
                else:
                    state_str += default_str
            state_str += "\n"
    return state_str

"""actions:
0 - move
1 - turnLeft
2 - turnRight
3 - pickMarker
4 - putMarker
5 - finish
[[walls], [markers], [marker_post], avatar_row, avatar_col, avatar_or, avatar_row_post, avatar_col_post, avatar_or_post]
"""

class Karel_Environment:
    r"""Environment that samples initial states and models transitions.
    """
    def __init__(self, num_karel_tasks:int, filepath:str, num_rows:int, num_cols:int, heuristic_rewards:bool=False,
        reward_default:float = -0.1,
        reward_success:float = 10.0,
        reward_unecessary_action:float = -1.0,
        reward_necessary_action:float = 1.0,
        reward_crash:float = 0.0) -> None:
        r"""Operates on a single state and action.
        :param num_karel_tasks (int): Number of Karel tasks in filepath.
        :param filepath (string): Directory, where sequences and tasks are stored in .json format in the respective folders. 
        :param num_rows (int): Number of rows in the grid.
        :param num_cols (int): Number of columns in the grid.
        :param heuristic_rewards (bool): If true, rewards are calculated based on a heuristic.
        :param reward_default (float): Reward for default action.
        :param reward_success (float): Reward for success (successfully reached the post-grid).
        :param reward_unecessary_action (float): Reward for unecessary marker actions (i.e. unecessarily picking up or placing markers).
        :param reward_necessary_action (float): Reward for necessary marker actions (i.e. picking up or placing markers, such that they match the post-grid).
        :param reward_crash (float): Reward for crashing the Karel task or finishing in state that is not the post-grid.
        """
        self.num_karel_tasks = num_karel_tasks
        self.filepath = filepath
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.heuristic_rewards = heuristic_rewards
        self.reward_default = reward_default
        self.reward_success = reward_success
        self.reward_unecessary_action = reward_unecessary_action
        self.reward_necessary_action = reward_necessary_action
        self.reward_crash = reward_crash
        

    def sample_task(self) -> Tuple[list, np.ndarray]:
        idx = random.randint(0, self.num_karel_tasks-1)
        seq_path = self.filepath + "/seq/" + str(idx) + "_seq.json"
        task_path = self.filepath + "/task/" + str(idx) + "_task.json"
        num_rows, num_cols, vec =  load_task(task_path)
        if(num_rows != self.num_rows or num_cols != num_cols):
            raise ValueError(f"Dimension of Karel Task did not match environment values: expected rows={self.num_rows}, expected cols={self.num_cols}; got: {num_cols}, {num_rows}")
        return load_seq(seq_path), vec


    def transition(self, vec:np.ndarray, action:int) -> Tuple[np.ndarray, float, bool]:
        r"""Models the transition from one state to the next state.
        Returns the reward and the next state.
        """
        vec_cop = vec.copy()
        vec_cop2 = vec.copy()
        next_state, terminal = self.get_next_state(vec_cop, action)
        if self.heuristic_rewards:
            reward = self.get_heuristic_reward(vec_cop2, next_state, action, terminal)
        else:
            reward = self.get_reward(vec_cop2, action)
        return next_state, reward, terminal
    
    def get_next_state(self, vec:np.ndarray, action:int) -> Tuple[np.ndarray, bool]:
        size = self.num_rows * self.num_cols
        avatar_row = vec[3*size]
        avatar_col = vec[3*size+1]
        terminal = False

        if action == 0:
            avatar_or = vec[3*size+2]
            if avatar_or == 0: # NORTH
                if avatar_row - 1 < 0: # run out of grid -> crash
                    terminal = True
                elif vec[self.num_cols * (avatar_row - 1) + avatar_col] == 1: # run into wall -> crash
                    terminal = True
                else:
                    vec[3*size] = avatar_row - 1
            elif avatar_or == 1: # EAST
                if avatar_col + 1 >= self.num_cols: # run out of grid -> crash
                    terminal = True
                elif vec[self.num_cols * avatar_row + (avatar_col + 1)] == 1: # run into wall -> crash
                    terminal = True
                else:
                    vec[3*size+1] = avatar_col + 1
            elif avatar_or == 2: # SOUTH
                if avatar_row + 1 > self.num_rows: # run out of grid -> crash
                    terminal = True
                elif vec[self.num_cols * (avatar_row + 1) + avatar_col] == 1: # run into wall -> crash
                    terminal = True
                else:
                    vec[3*size] = avatar_row + 1
            elif avatar_or == 3: # WEST
                if avatar_col - 1 < 0: # run out of grid -> crash
                    terminal = True
                elif vec[self.num_cols * avatar_row + (avatar_col - 1)] == 1: # run into wall -> crash
                    terminal = True
                else:
                    vec[3*size+1] = avatar_col - 1     
        elif action == 1:
            vec[3 * size + 2] = (vec[3 * size + 2] - 1) % 4
        elif action == 2:
            vec[3 * size + 2] = (vec[3 * size + 2] + 1) % 4
        elif action == 3:
            if vec[self.num_cols * avatar_row + avatar_col + size] == 0: # marker not there -> crash
                terminal = True
            else:
                vec[self.num_cols * avatar_row + avatar_col + size] = 0
        elif action == 4:
            if vec[self.num_cols * avatar_row + avatar_col + size] == 1: # marker there -> crash
                terminal = True
            else:
                vec[self.num_cols * avatar_row + avatar_col + size] = 1
        elif action == 5:
            terminal = True
        
        return vec, terminal

    def get_reward(self, vec:np.ndarray, action:int) -> float:

        size = self.num_rows * self.num_cols
        avatar_row = vec[3*size]
        avatar_col = vec[3*size+1]
        
        if action == 0:
            avatar_or = vec[3*size+2]
            if avatar_or == 0: # NORTH
                if avatar_row - 1 < 0: # run out of grid -> crash
                    return self.reward_crash
                elif vec[self.num_cols * (avatar_row - 1) + avatar_col] == 1: # run into wall -> crash
                    return self.reward_crash
                else:
                    return self.reward_default
            elif avatar_or == 1: # EAST
                if avatar_col + 1 >= self.num_cols: # run out of grid -> crash
                    return self.reward_crash
                elif vec[self.num_cols * avatar_row + (avatar_col + 1)] == 1: # run into wall -> crash
                    return self.reward_crash
                else:
                    return self.reward_default
            elif avatar_or == 2: # SOUTH
                if avatar_row + 1 > self.num_rows: # run out of grid -> crash
                    return self.reward_crash
                elif vec[self.num_cols * (avatar_row + 1) + avatar_col] == 1: # run into wall -> crash
                    return self.reward_crash
                else:
                    return self.reward_default
            elif avatar_or == 3: # WEST
                if avatar_col - 1 < 0: # run out of grid -> crash
                    return self.reward_crash
                elif vec[self.num_cols * avatar_row + (avatar_col - 1)] == 1: # run into wall -> crash
                    return self.reward_crash
                else:
                    return self.reward_default
        elif action == 3:
            if vec[self.num_cols * avatar_row + avatar_col + size] == 0: # marker not there -> crash
                return self.reward_crash
            elif vec[self.num_cols * avatar_row + avatar_col + 2*size] == 1: # marker in postgrid
                return self.reward_unecessary_action
            else: # marker not in postgrid
                return self.reward_necessary_action
        elif action == 4:
            if vec[self.num_cols * avatar_row + avatar_col + size] == 1: # marker already there -> crash
                return self.reward_crash
            elif vec[self.num_cols * avatar_row + avatar_col + 2*size] == 0: # marker not in postgrid
                return self.reward_unecessary_action
            else: # marker in postgrid
                return self.reward_necessary_action
        elif action == 5:
            if np.array_equal(vec[size:2*size], vec[2*size:3*size]) and np.array_equal(vec[3*size:3*size+3], vec[3*size+3:3*size+6]):
                return self.reward_success
            else:
                return self.reward_crash
        else:
            return self.reward_default
        

    def get_heuristic_reward(self, state:np.ndarray, next_state:np.ndarray, action:int, terminal:bool) -> float:
        reward_dist_improv = 0.1
        reward_marker_improv = 1.0
        reward_sucess = 10.0
        reward_default = -0.01
        reward_crash = -0.1
        size = self.num_rows * self.num_cols

        if terminal:
            if action == 5:
                if np.array_equal(state[size:2*size], state[2*size:3*size]) and np.array_equal(state[3*size:3*size+3], state[3*size+3:3*size+6]):
                    return reward_sucess
                else:
                    return reward_crash
            else:
                return reward_crash

        markers_diff = np.abs(state[size:2*size] - state[2*size:3*size], dtype=int)
        if np.any(markers_diff): # there are positions, where markers must be changed
            new_markers_diff = np.abs(next_state[size:2*size] - next_state[2*size:3*size], dtype=int)
            if np.sum(new_markers_diff) > np.sum(markers_diff):
                return -reward_marker_improv
            elif np.sum(new_markers_diff) < np.sum(markers_diff):
                return reward_marker_improv
            
            pos = np.array([[i//self.num_cols, i%self.num_cols] for i in range(size)])
            markers_pos = pos[markers_diff == 1]
            current_min_dist = np.min(np.linalg.norm(markers_pos - state[3*size:3*size+2] , 1, axis= 1))
            next_min_dist = np.min(np.linalg.norm(markers_pos - next_state[3*size:3*size+2] , 1, axis= 1))
            if next_min_dist < current_min_dist:
                return reward_dist_improv
            elif next_min_dist > current_min_dist:
                return -reward_dist_improv
            else:
                return reward_default

        else: # no markers must be changed
            if action == 3 or action == 4:
                return -reward_marker_improv
            current_dist = np.linalg.norm(state[3*size:3*size+2] - state[3*size+3:3*size+5])
            next_dist = np.linalg.norm(next_state[3*size:3*size+2] - next_state[3*size+3:3*size+5])
            if next_dist < current_dist:
                return reward_dist_improv
            elif next_dist > current_dist:
                return -reward_dist_improv
            else:
                return reward_default
       