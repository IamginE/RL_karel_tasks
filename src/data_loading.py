import json
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset

def read_json(filepath:str) -> dict:
    f = open(filepath)
    data = json.load(f)
    f.close
    return data

def action_to_int(action:str) -> int:
    if action == "move":
        return 0
    elif action == "turnLeft":
        return 1
    elif action == "turnRight":
        return 2
    elif action == "pickMarker":
        return 3
    elif action == "putMarker":
        return 4
    elif action == "finish":
        return 5

def dir_to_num(dir:str) -> int:
    if dir == "north":
        return 0
    elif dir == "east":
        return 1
    elif dir == "south":
        return 2
    elif dir == "west":
        return 3

# state vectors look like this: 
# [[walls], [markers], [marker_post], avatar_row, avatar_col, avatar_or, 
# avatar_row_post, avatar_col_post, avatar_or_post, flag_terminal]
def state_to_vec(state_dict:dict) -> Tuple[int, int, np.ndarray]:
    r""" Transforms the given dictionary representing a state parsed from a .json file to its vectorized representation.
    :param state_dict: Dictionary representing a state parsed from a .json file.
    :return: Tuple of the form (num_rows, num_cols, state_vector).
    """

    # initialize state vector
    num_rows = state_dict["gridsz_num_rows"]
    num_cols = state_dict["gridsz_num_cols"]

    size = num_cols * num_rows
    vec = np.zeros(size * 3 + 6, dtype=int)

    for pos in state_dict["walls"]:
        vec[num_cols * pos[0] + pos[1]] = 1

    for pos in state_dict["pregrid_markers"]:
        vec[num_cols * pos[0] + pos[1] + size] = 1

    for pos in state_dict["postgrid_markers"]:
        vec[num_cols * pos[0] + pos[1] + 2*size] = 1

    vec[3 * size] = state_dict["pregrid_agent_row"]
    vec[3 * size + 1] = state_dict["pregrid_agent_col"]
    vec[3 * size + 2] = dir_to_num(state_dict["pregrid_agent_dir"])

    vec[3 * size + 3] = state_dict["postgrid_agent_row"]
    vec[3 * size + 4] = state_dict["postgrid_agent_col"]
    vec[3 * size + 5] = dir_to_num(state_dict["postgrid_agent_dir"])
    return num_rows, num_cols, vec

def load_task(filepath:str) -> Tuple[int, int, np.ndarray]:
    return state_to_vec(read_json(filepath))

def load_seq(filepath:str) -> list:
    return read_json(filepath)["sequence"]

class Dataset_Supervision(Dataset):
    r"""
    Custom pytorch Dataset class that loads vectorized states and the corresponding action from a csv.
    Extends the class torch.utils.data.Dataset.
    """
    def __init__(self, csv_file:str, vec_size:int, seperator = ',') -> None:
        r"""
        :param csv_file (string): Path to the csv file with data.
        :param vec_size (int): Size of a vecotrized representation of a state.
        :param sperator (string): Seperator used in the csv file.
        """
        self.inputs = np.genfromtxt(csv_file, dtype=float, delimiter=seperator, usecols=[i for i in range(vec_size)])
        self.targets = np.genfromtxt(csv_file, dtype=int, delimiter=seperator, usecols=[vec_size])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return torch.from_numpy(self.inputs[idx]).to(torch.float32),  self.targets[idx]
