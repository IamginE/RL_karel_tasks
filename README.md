# Deep RL with PPO for Karel Tasks

This project solves 4-by-4 Karel Tasks as specified in `project.pdf` using deep RL.
Concretely, this project solves this via a combination of imitation learning and a variant of the PPO-Algorithm.

## Project Structure

### Folders
- The `data` folder contains training and validation dataset in the defined `.json` format. Additionally, it contains datasets in `.csv` format that can be used for supervised training. They are created from the optimal sequences given in `data/train/seq` in combination with the corresponding states obtained by executing the given actions on the corresponding task in `data/train/task`.
  - `supervised_first_<num>.csv` contains data generated from the first `<num>` tasks in `data/train/task`. For example, `supervised_first_50.csv` contains data obtained from tasks with id 0 to 49.
  - `supervised_full.csv` contains a dataset for supervised training obtained from all tasks in `data/train/task`.
- The `logs` folder contains logs that can be obtained, when comparing models, e.g., with `test_params` from `src/imitation_learning.py`.
- The `saved_models` folder contains parameters for networks that were trained and saved afterwards.
- The `plots` folder contains all generated plots, e.g., `test_params` from `src/imitation_learning.py` also produces plots in addition to the logs.
- The `src` folder contains all python source code.

### Python Files
- `create_supervised_data.py` contains a single function that uses an environment to generate datasets that can be used for supervised training (imitation learning) and saves them at the specified location.
- `data_loading.py` handles all data loading tasks. This includes reading from the `.json` files. It has functions that produce the state feature presentation as specified in `project2_train.pdf`. Additionally, it contains the Dataset that handles data flow during supervised training.
- `environment.py` contains a function for printing a visuliaziation of a vectorized state and the `Karel_Environment` that handles sampling initial states, as well as calculating state transitions and rewards.
- `evaluation.py` contains a function used to execute and evaluate policies on multiple Karel tasks and computes the number of tasks solved as well as the average return obtained.
- `execute_policy.py` contains functions used to print out the actions from a policy executed on a given Karel task, e.g., prints `["move", "turnRight", "fiish"]` to the console.
- `imitation_learning.py` contains functions to 1) pretrain a model in a supervised way and save it, 2) test optimzation of supervised learning on the same dataset for different learning rates (with SGD). 2) produces plots and logs.
- `networks.py` contains actor (policy) and critic (value) network.
- `plot.py` contains all plotting functions used for evaluation.
- `rollout_buffer.py` uses a rollout buffer that is heavily inspired by https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py. This buffer is used for storing the data produced in episodes generated when training with PPO. It also contains the routines to compute the advantage values.
- `trainer.py` cotains classes handling training. The `PPO_Trainer` applies the PPO-Algorithm, while the `FF_trainer` is used for supervised training. Additionally, the trainer compute some performance metrics.
- `test_env.py` contains a very simplistic environment for sanity checking the PPO-Algorithm (see below). 

## Test Environment
The test environment consists of 5 non-terminal states that are arranged in a straight line with two terminal states, one at each side: `(goal) - (s1) - (s2) - (s3) - (s4) - (s5) - (goal)`
There are two actions: `move left` or `move right` that transition deterministically to the neighbour states, and the goal is to reach a terminal state using as few moves as possible.

## Reproducibility of results
All of the results reported in `project2_train.pdf` can be reproduced by uncommenting the corresponding sections in `reproduce_all.py` and running it from the main folder of the project.

I trained all models on CPU with a fixed random seed. However, since dataloading is different with GPU, the results may change when models are trained on GPU instead of CPU.

Please also note that some functions raise runtime errors, if they would overwrite a file that already exists. This means some sections of reproduce_all.py may cause a runtime error if certain files it tries to write already exist.

