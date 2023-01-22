import numpy as np
import torch

class Rollout_Buffer:
    r"""Buffer that stores epidata off an episode.
    Heavily inspired by: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
    """
    def __init__(self, buffer_size:int, state_size:int, device, gamma:float) -> None:
        r"""
        :param buffer_size (int): Maximum elements to be stored, i.e. maximum episode length.
        :param state_size (int): Length of the vectorized representation of a state.
        :param device: Pytorch device that is used for computation on Pytorch tensors.
        :param gamma (float): Discount factor.
        """
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.full = False
        self.pos = 0
        self.device = device
        self.gamma = gamma
        self.actions, self.rewards, self.deltas = None, None, None
        self.nstep_returns, self.states, self.values, self.probs = None, None, None, None
        self.reset()

    def reset(self) -> None:
        self.actions = np.zeros((self.buffer_size,), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.deltas = np.zeros((self.buffer_size,), dtype=np.float32)
        self.nstep_returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.states = np.zeros((self.buffer_size, self.state_size), dtype=np.int32)
        self.full = False
        self.pos = 0

    def add(self, state:np.ndarray, action:int, reward:float, prob:float, value:torch.Tensor) -> None:
        if self.full:
            raise RuntimeError("Buffer is full, can not add element.")
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.states[self.pos] = state.copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten().item()
        self.probs[self.pos] = prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def calculate_returns_deltas(self, n:int, last_value:torch.Tensor, is_cut_off:bool) -> None:
        r"""Calculates the n-step returns and the TD-error (delta) for each state in the buffer."""
        # calculate n-step returns and deltas
        nstep_return = 0
        for step in range(self.pos):
            if step + n < self.pos:
                nstep_return = self.values[step+n]
                for i in reversed(range(n)):
                    nstep_return = self.rewards[step+i] + self.gamma * nstep_return
            else:
                # At the end of an episode, we can use the value function of the last state, if it is a cut off state.
                # Otherwise, it is terminal and therfore the value will be 0.
                if is_cut_off:
                    nstep_return = last_value.clone().cpu().numpy().flatten().item() 
                else:
                    nstep_return = 0.0
                k = self.pos-1
                while k >= step:
                    nstep_return = self.rewards[k] + self.gamma * nstep_return
                    k -= 1
            self.nstep_returns[step] = nstep_return
            self.deltas[step] = nstep_return - self.values[step]

    def get(self) -> dict:
        r"""Returns a dictionary containing the data in the buffer."""
        return {
            "actions": self.actions[0:self.pos],
            "rewards": self.rewards[0:self.pos],
            "deltas": self.deltas[0:self.pos],
            "nstep_returns": self.nstep_returns[0:self.pos],
            "states": self.states[0:self.pos],
            "values": self.values[0:self.pos],
            "probs": self.probs[0:self.pos]
        }
    
    def __repr__(self) -> str:
        return f"""Rollout_Buffer(buffer_size={self.buffer_size}, state_size={self.state_size}, gamma={self.gamma}, full={self.full}, pos={self.pos},
states=\n{self.states},
actions=\n{self.actions},
rewards=\n{self.rewards},
values=\n{self.values},
probs=\n{self.probs},
nstep_returns=\n{self.nstep_returns},
deltas=\n{self.deltas})"""


