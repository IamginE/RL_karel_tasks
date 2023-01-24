from rollout_buffer import Rollout_Buffer

import torch.nn.functional as F
import torch
import numpy as np

from typing import Tuple

class FF_Trainer:
    r""" Trainer for a simple FF-NN that learns in a supervised way from input-output tuples.
    """
    def __init__(self, model, train_loader, eval_loader, optimizer, loss, device):
        self.trained_epochs = 0
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.loss = loss
        
    def train(self, log_interval) -> Tuple[float, float]:
        num_correct = 0
        total_loss = 0.0
        num_inputs = 0
        self.model.train()
        for batch_idx, (input, target) in enumerate(self.train_loader):
            input, target = input.to(self.device), target.to(self.device)
            self.optimizer.zero_grad() # zero gradients for every batch
            target = target.to(torch.long)
            output = self.model(input)
            loss = self.loss(output, target) 
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True).flatten()
            num_correct += (pred == target).sum().item()
            total_loss += loss.item() * input.size(0)
            num_inputs += input.size(0)
            

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    self.trained_epochs, batch_idx * len(input), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
        self.trained_epochs += 1
        total_loss /= float(num_inputs)
        accuracy = (num_correct / float(num_inputs)) * 100
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            total_loss, num_correct, len(self.train_loader.dataset), accuracy))
        return total_loss, accuracy
        
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        num_correct = 0
        total_loss = 0.0
        num_inputs = 0
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.eval_loader):
                input, target = input.to(self.device), target.to(self.device)
                target = target.to(torch.long)
                output = self.model(input)
                test_loss = self.loss(output, target)
                pred = output.argmax(dim=1, keepdim=True).flatten()
                num_correct += (pred == target).sum().item()
                total_loss += test_loss.item() * input.size(0)
                num_inputs += input.size(0)
            total_loss /= float(num_inputs)
            accuracy = (num_correct / float(num_inputs)) * 100
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            total_loss, num_correct, len(self.eval_loader.dataset), accuracy))
        return total_loss, accuracy

class PPO_Trainer:
    r""" Trainer for an PPO agent that does the training and evaluation loop.
    """
    def __init__(self, actor, critic, cut_off:int, environment, gamma:float,
                state_size:int, num_episodes:int, n_steps:int, 
                clip_epsilon:float, k:int, device, optimizer_actor, optimizer_critic) -> None:         
        self.trained_epochs = 0
        self.cut_off = cut_off
        self.environment = environment
        self.num_episodes = num_episodes
        self.n_steps = n_steps
        self.clip_epsilon = clip_epsilon
        self.k = k
        self.actor = actor
        self.critic = critic
        self.device = device
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.rollout_buffer = Rollout_Buffer(buffer_size=cut_off, state_size=state_size, device=device, gamma=gamma)
    
    def sample_action(self, probs) -> int:
        r""" Sample an action from the given choices and probabilities.
        """
        return np.random.choice(np.arange(len(probs)), p=probs)

    def generate_rollout(self) -> None:
        r""" Generate a rollout for the current policy network.
        """
        self.rollout_buffer.reset()
        _, state = self.environment.sample_task()  
        
        with torch.no_grad():
            terminal = False
            for _ in range(self.cut_off):
                state_model = np.array([state])
                action_probs = self.actor(torch.from_numpy(state_model).float().to(self.device))
                value = self.critic(torch.from_numpy(state_model).float().to(self.device))
                action_probs = action_probs.clone().cpu().numpy()
                action_probs = action_probs[0]
                action = self.sample_action(action_probs)
                action_prob = action_probs[action]
                next_state, reward, terminal = self.environment.transition(state, action)
                self.rollout_buffer.add(state, action, reward, action_prob, value)
                state = next_state
                if terminal:
                    break
            last_value = self.critic(torch.from_numpy(state_model).float().to(self.device))
            self.rollout_buffer.calculate_returns_deltas(self.n_steps, last_value, not terminal)

    def train(self, epochs:int, log_interval:int) -> Tuple[list, list]:
        r""" Train the policy and critic networks for the given number of epochs.
        :returns: The losses of the actor and critic networks.
        """
        actor_losses = []
        critic_losses = []
        self.actor.train()
        self.critic.train()

        for epoch in range(epochs):
            actor_loss = 0.0
            states = []
            actions = []
            probs = []
            nstep_returns = []
            deltas = []
            for _ in range(self.num_episodes):
                self.generate_rollout()
                data = self.rollout_buffer.get()
                states.append(data["states"].copy())
                actions.append(data["actions"].copy())
                probs.append(data["probs"].copy())
                nstep_returns.append(data["nstep_returns"].copy())
                deltas.append(data["deltas"].copy())
            states = np.concatenate(states)
            actions = np.concatenate(actions)
            probs = np.concatenate(probs)
            nstep_returns = np.concatenate(nstep_returns)
            deltas = np.concatenate(deltas)
            states = torch.tensor(states).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            probs = torch.tensor(probs).to(self.device)
            nstep_returns = torch.tensor(nstep_returns).to(self.device)
            deltas = torch.tensor(deltas).to(self.device)
            states = states.float()
            
            self.optimizer_critic.zero_grad()
            for _ in range(self.k):
                self.optimizer_actor.zero_grad()
                new_probs = self.actor(states)
                new_probs = new_probs[torch.arange(new_probs.size(0)), actions.long()]
                
                # clipped actor surrogate loss
                probs_ratio = new_probs/probs
                probs_ratio_clipped = torch.clamp(probs_ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
                policy_loss_1 = deltas * probs_ratio
                policy_loss_2 = deltas * probs_ratio_clipped
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                policy_loss.backward()
                self.optimizer_actor.step()
                
                actor_loss += policy_loss.item()
            
            new_values = self.critic(states)
            new_values = new_values.flatten()
            # value loss 
            value_loss = F.mse_loss(nstep_returns, new_values)
            value_loss.backward()
            self.optimizer_critic.step()
            
            actor_losses.append(actor_loss/self.k)
            critic_losses.append(value_loss.item())
            if epoch % log_interval == 0:
                print('Train Epoch: {} \tValue Loss: {:.6f} \tActor Loss: {:.6f}'.format(self.trained_epochs, value_loss.item(), policy_loss.item()))

            self.trained_epochs += 1

        return actor_losses, critic_losses
