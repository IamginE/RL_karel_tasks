import numpy as np
from typing import Tuple
import random
import torch
import torch.optim as optim

from trainer import PPO_Trainer
from networks import Policy_Network, Value_Network
"""actions:
0 - move left
1 - move right
"""

class Simple_Test_Env:
    r"""Environment for testing (goal) - (s1) - (s2) - (s3) - (s4) - (s5) - (goal).
    """
    def __init__(self) -> None:
        self.reward_default = 0.0
        self.reward_success = 10.0
       

    def sample_task(self) -> Tuple[list, np.ndarray]:
        i = random.randint(0, 4)
        vec = np.zeros(5, dtype=int)
        vec[i] = 1
        return [], vec


    def transition(self, vec:np.ndarray, action:int) -> Tuple[np.ndarray, float, bool]:
        r"""Models the transition from one state to the next state.
        Returns the reward and the next state.
        """
        vec_cop = vec.copy()
        terminal = False
        reward = self.reward_default
        next_state = np.zeros(5, dtype=int)
        index = np.where(vec_cop == 1)[0][0]
        if action == 0:
            if index == 0:
                next_state = vec_cop
                terminal = True
                reward = self.reward_success
            else:
                next_state[index - 1] = 1
        elif action == 1:
            if index == 4:
                next_state = vec_cop
                terminal = True
                reward = self.reward_success
            else:
                next_state[index + 1] = 1
        return next_state, reward, terminal


def main():
    # Training Parameters
    env = Simple_Test_Env()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    actor = Policy_Network(5, 2, False)
    actor.to(device)

    gamma = 0.95

    actor.set_softmax(True)

    critic = Value_Network(5, 1).to(device)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.003) 
    optimizer_actor_2 = optim.Adam(actor.parameters(), lr=0.001) 

    _, state = env.sample_task()
    print(state)
    print(actor(torch.tensor(state).to(device).float()))
    print(critic(torch.tensor(state).to(device).float()))
    
    cut_off = int((5/(1-gamma)))
    ppo_trainer = PPO_Trainer(actor,
        critic, 
        cut_off=cut_off, 
        environment=env,
        num_episodes=32,
        gamma=gamma, 
        state_size=5, 
        n_steps=5, 
        clip_epsilon=0.2, 
        k=2, 
        device=device, 
        optimizer_actor=optimizer_actor_2,
        optimizer_critic=optimizer_critic
    )
    ppo_trainer.train(300, 50)
    
    print(actor(torch.tensor(state).to(device).float()))
    print(critic(torch.tensor(state).to(device).float())) 
    for _ in range(10):
        _, state = env.sample_task()  
        print(state)
        print(actor(torch.tensor(state).to(device).float()))
        print(critic(torch.tensor(state).to(device).float())) 
main()
