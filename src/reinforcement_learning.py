import torch
import numpy as np
import random
import os

from environment import Karel_Environment
from trainer import PPO_Trainer
from plot import plot
from evaluation import eval_policy

from networks import Policy_Network, Value_Network
import torch.optim as optim
from torch import nn


def test_params_rl(train_data_path:str, test_data_path:str, 
    logging_path:str, id:str, seed:int, env, epochs:int, 
    gamma:float, lr_actor:float, lr_critic:float, eval_interval:int, device, 
    vec_size:int=4*4*3+6, num_episodes:int=32, k:int=5, n_steps:int=5, clip_epsilon:float=0.2,
    load_actor=True, load_critic=False, actor_path:str="", critic_path:str="",
    save_actor=False, save_critic=False, actor_path_out:str="", critic_path_out:str="") -> None:
    r"""
    Trains actor (policy) network and critic (value) network via PPO for the given parameters, logs performance metrics and plots them in a .png file.
    If load values are True, the networks are loaded from the given paths and trained for the given number of epochs. Otherwise, the networks are trained from scratch.
    :param train_data_path (str): Path to folder with training data.
    :param test_data_path (str): Path to folder with testing data.
    :param logging_path (str): Path to folder, where logs are written.
    :param id (str): Identifier for the plots and logging, prefix of the respective file names.
    :param seed (int): Seed used for fixing random seeds.
    :param env: Environment used for training.
    :param epochs (int): Number of epochs to train.
    :param gamma (float): Discount factor.
    :param lr_actor (float): Learning rate for the actor (policy) network.
    :param lr_critic (float): Learning rate for the critic (value) network.
    :param eval_interval (int): Interval for evaluating the performance of the networks.
    :param device: Device used in training (cpu or gpu).
    :param vec_size (int): Size of a vector representing a state.
    :param num_episodes (int): Number of episodes to train on per epoch.
    :param k (int): Number of gradient steps for the actor (policy) network per epoch.
    :param n_steps (int): Number of steps for the n-step TD returns.
    :param clip_epsilon (float): Epsilon value for clipping the PPO objective.
    :param load_actor (bool): If True, the actor network is loaded from the given path.
    :param load_value (bool): If True, the value network is loaded from the given path.
    :param actor_path (str): Path to the actor network.
    :param value_path (str): Path to the value network.
    :param save_actor (bool): If True, the actor network is saved to the given path.
    :param save_value (bool): If True, the value network is saved to the given path.
    :param actor_path_out (str): Path to save the actor network.
    :param value_path_out (str): Path to save the value network.
    """
    # indices for logging
    train_min = 0
    train_max = 24000
    test_min = 100000
    test_max = 102400

    # fix random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # load models 
    actor = Policy_Network(54, 6, False)
    critic = Value_Network(54, 1)
    if load_actor:
        checkpoint_actor = torch.load(actor_path)
        actor.load_state_dict(checkpoint_actor['model_state_dict'])
    if load_critic:
        checkpoint_critic = torch.load(critic_path)
        critic.load_state_dict(checkpoint_critic['model_state_dict'])
    
    actor.set_softmax(True)
    actor.to(device)
    critic.to(device)
    
    # optimizers
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    
    # initialize trainer
    cut_off = int((5/(1-gamma)))
    ppo_trainer = PPO_Trainer(actor,
        critic, 
        cut_off=cut_off, 
        environment=env,
        num_episodes=num_episodes,
        gamma=gamma, 
        state_size=vec_size, 
        n_steps=n_steps, 
        clip_epsilon=clip_epsilon, 
        k=k, 
        device=device, 
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic
    )
    
    logging = logging_path + "/" + id +  "_log_ppo_solution" + ".csv"
    if os.path.isfile(logging):
        raise RuntimeError("CSV file with the same name exists: {}".format(logging))
    
    logging_loss = logging_path + "/" + id +  "_log_ppo_solution_loss" + ".csv"
    if os.path.isfile(logging):
        raise RuntimeError("CSV file with the same name exists: {}".format(logging_loss))

    actor_losses = []
    critic_losses = []
    
    with open(logging, 'w') as fp:
        fp.write("Epoch, Average Return Train, Num Shortest Train, Num Solved Train, Average Return Test, Num Shortest Test, Num Solved Test\n")
        avg_return_train, num_shortest_train, num_solved_train = eval_policy(actor, train_data_path, train_min, train_max, 30, 4, 4, gamma, device)
        avg_return_test, num_shortest_test, num_solved_test = eval_policy(actor, test_data_path, test_min, test_max, 30, 4, 4, gamma, device)
        out_str = "{}, {:.4f}, {}, {}, {:.4f}, {}, {}\n".format(0, avg_return_train, num_shortest_train, num_solved_train, avg_return_test, num_shortest_test, num_solved_test)
        fp.write(out_str)

        num_evals = epochs // eval_interval
        for i in range(num_evals):
            actor_loss, critic_loss = ppo_trainer.train(eval_interval, 50)  
            actor_losses = actor_losses + actor_loss
            critic_losses = critic_losses + critic_loss
            avg_return_train, num_shortest_train, num_solved_train = eval_policy(actor, train_data_path, train_min, train_max, 30, 4, 4, gamma, device)
            avg_return_test, num_shortest_test, num_solved_test = eval_policy(actor, test_data_path, test_min, test_max, 30, 4, 4, gamma, device)
            out_str = "{}, {:.4f}, {}, {}, {:.4f}, {}, {}\n".format((i+1)*eval_interval, avg_return_train, num_shortest_train, num_solved_train, avg_return_test, num_shortest_test, num_solved_test)
            fp.write(out_str)

    if save_actor:
        torch.save({
            'model_state_dict': actor.state_dict(),
            'optimizer_state_dict': optimizer_actor.state_dict()
            }, actor_path_out)
    if save_critic:
        torch.save({
            'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': optimizer_critic.state_dict()
            }, critic_path_out)

    with open(logging_loss, 'w') as fp:
        fp.write("Epoch, Average Actor Loss, Critic Loss\n")
        for i in range (len(actor_losses)):
            out_str = "{}, {:.4f}, {:.4f}\n".format(i, actor_losses[i], critic_losses[i])
            fp.write(out_str)
    
