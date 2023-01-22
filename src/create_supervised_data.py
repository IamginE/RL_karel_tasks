from environment import Karel_Environment
from data_loading import load_seq, load_task, action_to_int

import os

def generate_supervised_data(start_idx:int, end_idx:int, path:str, out:str) -> None:
    r""" Generates a .csv file with training data for a supervised learning task,
    where the goal is to predict an action (target) from the given task.
    To generate the data Karel tasks with id in the index range [start_idx, end_idx) are used.
    """
    if os.path.isfile(out):
        raise RuntimeError("File with the same name exists: {}".format(out))
    env = Karel_Environment(-1, "", 4, 4) # we don't need to samle initial states here
    with open(out, 'w') as fp:
        for i in range(start_idx, end_idx, 1):
            task_file = path + "/task/" + str(i) + "_task.json"
            seq_file = path + "/seq/" + str(i) + "_seq.json"
            _, _, vec = load_task(task_file)
            shortest_seq = load_seq(seq_file)
            for action in shortest_seq:
                out_str = ""
                for i in vec:
                    out_str += str(i) + ","
                int_action = action_to_int(action)
                fp.write(out_str + str(int_action) + "," + action + "\n")
                vec, _, _ = env.transition(vec, int_action)





