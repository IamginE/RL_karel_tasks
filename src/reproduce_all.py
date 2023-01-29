from create_supervised_data import generate_supervised_data
from imitation_learning import test_params, pretrain
from networks import Policy_Network
from data_loading import Dataset_Supervision
from trainer import FF_Trainer
from evaluation import eval_policy
from reinforcement_learning import test_params_rl
from plot import plot
from environment import Karel_Environment

import torch
import numpy as np
import random
from torch import optim
from torch import nn

# random seed used in all experiments
SEED =  1337

# Uncomment the following lines to generate the data for supervised learning
"""
generate_supervised_data(0, 50, "data/train", "data/supervised_first_50.csv") 
generate_supervised_data(0, 100, "data/train", "data/supervised_first_100.csv") 
generate_supervised_data(0, 4000, "data/train", "data/supervised_first_4000.csv") 
generate_supervised_data(0, 24000, "data/train", "data/supervised_full.csv") 
generate_supervised_data(100000, 102400, "data/val", "data/supervised_val.csv")
"""

# Test supervised learning on the full training data set to verify model capacity and to get a baseline (I.2.c))
# Uncomment the following lines to run the tests
"""
test_params("data/supervised_full.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "capacity",
            [0.3, 0.1, 0.05, 0.03],
            SEED,
            20,
            [2*i for i in range(1, 11)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""

# Produce and save baseline model
"""
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = Policy_Network(54, 6, False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pretrain("./data/supervised_full.csv",
    "./saved_models/actor_pretrained_full.pt",
    64,
    25,
    nn.CrossEntropyLoss(),
    model,
    optim.SGD(model.parameters(), lr=0.1),
    device,
    100
    )
"""

# evaluate the baseline model
# only prints results to the console
"""
train_batch_size = 64
test_batch_size = 64
train_kwargs = {'batch_size': train_batch_size}
test_kwargs = {'batch_size': test_batch_size}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
  
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
  
training_data = Dataset_Supervision(csv_file="data/supervised_full.csv", vec_size=4*4*3+6)
test_data = Dataset_Supervision(csv_file="data/supervised_val.csv", vec_size=4*4*3+6)
train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_full.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.to(device)
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()

gamma = 0.99

actor.set_softmax(True)
eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)
"""

# Test supervised learning for different amount of praining data. 
# The main purpose is to find a good learning rate and a good stopping point for the training.

"""
# first 50
test_params("data/supervised_first_50.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "first_50",
            [0.3, 0.1, 0.05, 0.03],
            SEED,
            200,
            [10*i for i in range(0, 21)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# first 100
test_params("data/supervised_first_100.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "first_100",
            [0.3, 0.1, 0.05, 0.03],
            SEED,
            100,
            [5*i for i in range(0, 21)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# first 4000
test_params("data/supervised_first_4000.csv",
            "data/supervised_val.csv",
            "plots",
            "logs",
            "first_4000",
            [0.3, 0.1, 0.05, 0.03],
            SEED,
            50,
            [2*i for i in range(0, 26)],
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""


# Produce and save starting models trained on first few Karel tasks
"""
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Produce and save starting model trained on first 50 Karel tasks
model = Policy_Network(54, 6, False)
model.to(device)

pretrain("./data/supervised_first_50.csv",
    "./saved_models/actor_pretrained_first_50.pt",
    64,
    100,
    nn.CrossEntropyLoss(),
    model,
    optim.SGD(model.parameters(), lr=0.1),
    device,
    100
    )

# Produce and save starting model trained on first 100 Karel tasks
model = Policy_Network(54, 6, False)
model.to(device)

pretrain("./data/supervised_first_100.csv",
    "./saved_models/actor_pretrained_first_100.pt",
    64,
    100,
    nn.CrossEntropyLoss(),
    model,
    optim.SGD(model.parameters(), lr=0.1),
    device,
    100
    )

# Produce and save starting model trained on first 4000 Karel tasks
model = Policy_Network(54, 6, False)
model.to(device)

pretrain("./data/supervised_first_4000.csv",
    "./saved_models/actor_pretrained_first_4000.pt",
    64,
    20,
    nn.CrossEntropyLoss(),
    model,
    optim.SGD(model.parameters(), lr=0.1),
    device,
    100
    )
"""


# evaluate the starting model
# only prints results to the console
"""
train_batch_size = 64
test_batch_size = 64
train_kwargs = {'batch_size': train_batch_size}
test_kwargs = {'batch_size': test_batch_size}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
  
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
  
training_data = Dataset_Supervision(csv_file="data/supervised_full.csv", vec_size=4*4*3+6)
test_data = Dataset_Supervision(csv_file="data/supervised_val.csv", vec_size=4*4*3+6)
train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
gamma = 0.99

# model trained on first 50 tasks
actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_first_50.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.to(device)
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()

actor.set_softmax(True)
eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)

# model trained on first 100 tasks
actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_first_100.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.to(device)
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()

actor.set_softmax(True)
eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)

# model trained on first 4000 tasks
actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_pretrained_first_4000.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.to(device)
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()

actor.set_softmax(True)
eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)
"""

# Train the different pretrained models on the full training task set with PPO
"""
# model trained on first 50 tasks
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_50_2000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=200, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_50.pt", 
    critic_path="",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_50_2000.pt", 
    critic_path_out="./saved_models/critic_first_50_2000.pt")


# model trained on first 100 tasks
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=200, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_2000.pt", 
    critic_path_out="./saved_models/critic_first_100_2000.pt")


# model trained on first 4000 tasks
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_4000_2000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=200, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_4000.pt", 
    critic_path="",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_4000_2000.pt", 
    critic_path_out="./saved_models/critic_first_4000_2000.pt")
"""

# Plot the learning curves for PPO
"""
model_names=["pretrained first 50", "pretrained first 100", "pretrained first 4000"]
training_tasks_shortest = []
validation_tasks_shortest = []
training_tasks = []
validation_tasks = []
training_returns = []
validation_returns = []
for name in ["first_50_2000_log_ppo_solution.csv", "first_100_2000_log_ppo_solution.csv", "first_4000_2000_log_ppo_solution.csv"]:
    data = np.loadtxt("logs/" + name, delimiter=",", skiprows=1)
    training_returns.append(data[:, 1])
    training_tasks_shortest.append(data[:, 2])
    training_tasks.append(data[:, 3])
    validation_returns.append(data[:, 4])
    validation_tasks_shortest.append(data[:, 5])
    validation_tasks.append(data[:, 6])


plot(training_returns, (0,10), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_avg_return_tr.png", 
    "Average return on the training dataset.", 
    "PPO training epochs", "average return", 
    linspace_mul=200,
    start_origin=True)
plot(training_tasks_shortest, (0,24000), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_shortest_sequences_tr.png", 
    "Number of tasks solved with the shortest sequence in the training dataset.", 
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)
plot(training_tasks, (0,24000), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_num_solved_tr.png", 
    "Number of tasks solved in the training dataset.", 
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)

plot(validation_returns, (0,10), [200*i for i in range(0,11)], model_names,
    "plots/ppo_avg_return_val.png",
    "Average return on the validation dataset.",
    "PPO training epochs", "average return", 
    linspace_mul=200,
    start_origin=True)
plot(validation_tasks_shortest, (0,2400), [200*i for i in range(0,11)], model_names,
    "plots/ppo_shortest_sequences_val.png",
    "Number of tasks solved with the shortest sequence in the validation dataset.",
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)
plot(validation_tasks, (0,2400), [200*i for i in range(0,11)], model_names,
    "plots/ppo_num_solved_val.png",
    "Number of tasks solved in the validation dataset.",
    "PPO training epochs", "number of tasks solved",
    linspace_mul=200, 
    start_origin=True)
"""

# train model with only reward for successfully solving a task
"""
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_zero",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False,
        reward_default=0.0,
        reward_success=10.0,
        reward_unecessary_action=0.0,
        reward_necessary_action=0.0,
        reward_crash=0.0),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=200, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_2000_zero.pt", 
    critic_path_out="./saved_models/critic_first_100_2000_zero.pt")
"""

# verify results on other seeds
"""
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_seed_42",
    seed=42, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")

test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_seed_69",
    seed=69, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")

test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_zero_seed_42",
    seed=42, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False,
        reward_default=0.0,
        reward_success=10.0,
        reward_unecessary_action=0.0,
        reward_necessary_action=0.0,
        reward_crash=0.0),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")

test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_zero_seed_69",
    seed=69, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False,
        reward_default=0.0,
        reward_success=10.0,
        reward_unecessary_action=0.0,
        reward_necessary_action=0.0,
        reward_crash=0.0),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")
"""

# test heuristic rewards
"""
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_heur",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, True),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=200, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_2000_heur.pt", 
    critic_path_out="./saved_models/critic_first_100_2000_heur.pt")
"""
# verify results on other seeds
"""
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_heur_seed_42",
    seed=42, 
    env=Karel_Environment(24000, "./data/train", 4, 4, True),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")

test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_heur_seed_69",
    seed=69, 
    env=Karel_Environment(24000, "./data/train", 4, 4, True),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=False, 
    actor_path="./saved_models/actor_pretrained_first_100.pt", 
    critic_path="",
    save_actor=False, 
    save_critic=False, 
    actor_path_out="", 
    critic_path_out="")
"""

# plot the learning curves for reward design
"""
model_names=["default reward design", "only reward on success", "heuristic reward design"]
training_tasks_shortest = []
validation_tasks_shortest = []
training_tasks = []
validation_tasks = []
training_returns = []
validation_returns = []
for name in ["first_100_2000_log_ppo_solution.csv", "first_100_2000_zero_log_ppo_solution.csv", "first_100_2000_heur_log_ppo_solution.csv"]:
    data = np.loadtxt("logs/" + name, delimiter=",", skiprows=1)
    training_returns.append(data[:, 1])
    training_tasks_shortest.append(data[:, 2])
    training_tasks.append(data[:, 3])
    validation_returns.append(data[:, 4])
    validation_tasks_shortest.append(data[:, 5])
    validation_tasks.append(data[:, 6])


plot(training_returns, (0,10), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_avg_return_rew_tr.png", 
    "Average return on the training dataset.", 
    "PPO training epochs", "average return", 
    linspace_mul=200,
    start_origin=True)
plot(training_tasks_shortest, (0,24000), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_shortest_sequences_rew_tr.png", 
    "Number of tasks solved with the shortest sequence in the training dataset.", 
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)
plot(training_tasks, (0,24000), [200*i for i in range(0,11)], model_names, 
    "plots/ppo_num_solved_rew_tr.png", 
    "Number of tasks solved in the training dataset.", 
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)

plot(validation_returns, (0,10), [200*i for i in range(0,11)], model_names,
    "plots/ppo_avg_return_rew_val.png",
    "Average return on the validation dataset.",
    "PPO training epochs", "average return", 
    linspace_mul=200,
    start_origin=True)
plot(validation_tasks_shortest, (0,2400), [200*i for i in range(0,11)], model_names,
    "plots/ppo_shortest_sequences_rew_val.png",
    "Number of tasks solved with the shortest sequence in the validation dataset.",
    "PPO training epochs", "number of tasks solved", 
    linspace_mul=200,
    start_origin=True)
plot(validation_tasks, (0,2400), [200*i for i in range(0,11)], model_names,
    "plots/ppo_num_solved_rew_val.png",
    "Number of tasks solved in the validation dataset.",
    "PPO training epochs", "number of tasks solved",
    linspace_mul=200, 
    start_origin=True)
"""

# evaluate model pretrained on first 100 tasks in a supverised manner
# only prints out the results to the console
"""
train_batch_size = 64
test_batch_size = 64
train_kwargs = {'batch_size': train_batch_size}
test_kwargs = {'batch_size': test_batch_size}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
  
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
  
training_data = Dataset_Supervision(csv_file="data/supervised_full.csv", vec_size=4*4*3+6)
test_data = Dataset_Supervision(csv_file="data/supervised_val.csv", vec_size=4*4*3+6)
train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
gamma = 0.99

# model trained on first 50 tasks
actor = Policy_Network(54, 6, False)
checkpoint_actor = torch.load("./saved_models/actor_first_100_2000.pt")
actor.load_state_dict(checkpoint_actor['model_state_dict'])
actor.to(device)
actor.set_softmax(False)

trainer_training_data = FF_Trainer(actor, None, train_loader, None, nn.CrossEntropyLoss(), device)
trainer_test_data = FF_Trainer(actor, None, test_loader, None, nn.CrossEntropyLoss(), device)

trainer_training_data.evaluate()
trainer_test_data.evaluate()
"""

# TRAIN THE AGENT TO BE A CHAMPION
"""
test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_4000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0002, 
    lr_critic=0.0005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=True, 
    actor_path="./saved_models/actor_first_100_2000.pt", 
    critic_path="./saved_models/critic_first_100_2000.pt",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_4000.pt", 
    critic_path_out="./saved_models/critic_first_100_4000.pt")


test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_6000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.0001, 
    lr_critic=0.00025, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=True, 
    actor_path="./saved_models/actor_first_100_4000.pt", 
    critic_path="./saved_models/critic_first_100_4000.pt",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_6000.pt", 
    critic_path_out="./saved_models/critic_first_100_6000.pt")


test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_8000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.00005, 
    lr_critic=0.0001, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=True, 
    actor_path="./saved_models/actor_first_100_6000.pt", 
    critic_path="./saved_models/critic_first_100_6000.pt",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_8000.pt", 
    critic_path_out="./saved_models/critic_first_100_8000.pt")

test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_10000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=2000, 
    gamma=0.99, 
    lr_actor=0.00001, 
    lr_critic=0.00005, 
    eval_interval=2000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=True, 
    actor_path="./saved_models/actor_first_100_8000.pt", 
    critic_path="./saved_models/critic_first_100_8000.pt",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_10000.pt", 
    critic_path_out="./saved_models/critic_first_100_10000.pt")


test_params_rl(train_data_path="./data/train",
    test_data_path="./data/val", 
    logging_path="./logs", 
    id="first_100_2000_11000",
    seed=SEED, 
    env=Karel_Environment(24000, "./data/train", 4, 4, False),
    epochs=1000, 
    gamma=0.99, 
    lr_actor=0.000001, 
    lr_critic=0.000005, 
    eval_interval=1000, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    vec_size=4*4*3+6, 
    num_episodes=32, 
    k=5, 
    n_steps=5, 
    clip_epsilon=0.2,
    load_actor=True, 
    load_critic=True, 
    actor_path="./saved_models/actor_first_100_10000.pt", 
    critic_path="./saved_models/critic_first_100_10000.pt",
    save_actor=True, 
    save_critic=True, 
    actor_path_out="./saved_models/actor_first_100_11000.pt", 
    critic_path_out="./saved_models/critic_first_100_11000.pt")
"""