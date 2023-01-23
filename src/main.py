from trainer import FF_Trainer, PPO_Trainer
from data_loading import Dataset_Supervision
from networks import Policy_Network, Value_Network
from environment import Karel_Environment
from data_loading import load_task
from evaluation import eval_policy

import torch
import torch.optim as optim
import numpy as np
import random
from torch import nn

def main():
    # Training Parameters
    epochs = 20 # 20 for first 4000, 120 for first 50
    train_batch_size = 64
    test_batch_size = 64
    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    learning_rate = 0.3 # 0.05

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
  
    training_data = Dataset_Supervision(csv_file="data/supervised_first_4000.csv", vec_size=4*4*3+6)
    test_data = Dataset_Supervision(csv_file="data/supervised_full.csv", vec_size=4*4*3+6)
    train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
  
    actor = Policy_Network(54, 6, False)
    actor.to(device)
    optimizer_actor = optim.SGD(actor.parameters(), lr=learning_rate)
    trainer = FF_Trainer(actor, train_loader, test_loader, optimizer_actor, nn.CrossEntropyLoss(), device)


    _, _, task1 = load_task("./data/train/task/0_task.json")
    _, _, task2 = load_task("./data/train/task/4_task.json")
    _, _, task3 = load_task("./data/train/task/14_task.json")
    state = np.array([task1, task2, task3], dtype=np.int32)

    for _ in range(epochs):
        trainer.train(100)
        trainer.evaluate()

    trainer.evaluate()

    gamma = 0.99

    actor.set_softmax(True)
    eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
    eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)

    critic = Value_Network(54, 1).to(device)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.0005) # 0.001, 0.00001, 0.0005
    optimizer_actor_2 = optim.Adam(actor.parameters(), lr=0.0002) # 0.0005, 0.000005, 0.0002

    # optimizer_critic = optim.SGD(critic.parameters(), lr=0.003) 
    # optimizer_actor_2 = optim.SGD(actor.parameters(), lr=0.0005) 
    print(actor(torch.tensor(state).to(device).float()))
    print(critic(torch.tensor(state).to(device).float()))
    
    cut_off = int((5/(1-gamma)))
    ppo_trainer = PPO_Trainer(actor,
        critic, 
        cut_off=cut_off, 
        environment=Karel_Environment(24000, "./data/train", 4, 4, False),
        num_episodes=64,
        gamma=gamma, 
        state_size=54, 
        n_steps=5, 
        clip_epsilon=0.2, 
        k=5, 
        device=device, 
        optimizer_actor=optimizer_actor_2,
        optimizer_critic=optimizer_critic
    )
    ppo_trainer.train(1000, 50)
    
    eval_policy(actor, "./data/train", 0, 24000, 30, 4, 4, gamma, device)
    eval_policy(actor, "./data/val", 100000, 102400, 30, 4, 4, gamma, device)
    print(actor(torch.tensor(state).to(device).float()))
    print(critic(torch.tensor(state).to(device).float()))
    actor.set_softmax(False)
    trainer.evaluate()
    

main()
