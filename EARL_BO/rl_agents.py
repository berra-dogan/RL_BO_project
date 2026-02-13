
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.optim as optim
from copy import deepcopy

# Define the ActorCritic neural network for the PPO agent
class ActorCritic(nn.Module):
    def __init__(self, num_state, num_action, action_std, device = "cpu"):
        super(ActorCritic, self).__init__()
        # Define the actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_action),
            nn.Tanh()
        )
        # Define the critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.device = device
        self.action_var = torch.full((num_action,), action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # Select an action based on the current policy
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action = action * (~torch.isnan(action))
        action = torch.clamp(action, -1., 1.)
        action_logprob = dist.log_prob(action)

        memory.states.append(state.cpu())
        memory.actions.append(action.cpu())
        memory.logprobs.append(action_logprob.cpu())

        return action.detach()

    def act_mean(self, state):
        # Select the mean action (used for evaluation)
        action_mean = self.actor(state)
        action_mean = action_mean * (~torch.isnan(action_mean))
        return action_mean

    def evaluate(self, state, action):
        # Evaluate the action and compute the value function
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# Define the PPO agent class
class PPO_Agent:
    def __init__(self, num_state, num_action, device = "cpu"):
        # Initialize PPO agent parameters

        self.device = device

        self.num_state = num_state
        self.num_action = num_action
        self.learning_rate = 0.001
        self.action_std = 0.1
        self.K_epochs = 100
        self.eps_clip = 0.2
        self.betas = (0.9, 0.999)
        self.action_decay = 0.99
        self.gamma = 0.95
        self.gamma_increase = 1
        self.VF_coeff = 0.5
        self.Entropy_coeff = 0.01
        self.action_std_min = 0.01
        self.freeze_num = 2

        # Initialize the policy and old policy networks
        self.policy = ActorCritic(self.num_state, self.num_action, self.action_std).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, betas=self.betas)
        self.policy_old = ActorCritic(self.num_state, self.num_action, self.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, state, memory):
        # Get an action from the current policy
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        else:
            state = state.to(self.device)
        action = self.policy_old.act(state, memory).cpu().data.numpy().squeeze()
        return action.reshape(-1)

    def get_action_eval(self, state):
        # Get an action for evaluation (using the mean of the policy)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        else:
            state = state.to(self.device)
        return self.policy_old.act_mean(state).cpu().data.numpy().squeeze()

    def update(self, memory, encoder_optimizer):
        # Update the policy using PPO
        self.action_std = max(self.action_std * self.action_decay, self.action_std_min)
        # Compute returns and advantages
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # Convert stored experience to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack([action.reshape(-1) for action in memory.actions]).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.VF_coeff * self.MseLoss(state_values,
                                                                           rewards) - self.Entropy_coeff * dist_entropy

            self.optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            encoder_optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.action_var = torch.full((self.num_action,), self.action_std * self.action_std).to(self.device)
        self.policy_old.action_var = torch.full((self.num_action,), self.action_std * self.action_std).to(self.device)
        self.gamma = 1 - self.gamma_increase * (1 - self.gamma)

    def update_initial(self, memory):
        # Initial update of the policy (used for off-policy learning)
        # Compute returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).to(self.device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device)).detach()
        # Optimize policy
        for _ in range(1000):
            _, state_values, _ = self.policy.evaluate(old_states, old_actions)
            actions = self.policy.act_mean(old_states)

            loss = self.MseLoss(state_values, rewards) + self.MseLoss(actions, old_actions)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy.action_var = torch.full((self.num_action,), self.action_std * self.action_std).to(self.device)
        self.policy_old.action_var = torch.full((self.num_action,), self.action_std * self.action_std).to(self.device)
        self.gamma = 1 - self.gamma_increase * (1 - self.gamma)

    def transfer_learning(self, policy_file_num):
        # Store the initial policy parameters in memory rather than on disk
        initial_policy_params = deepcopy(self.policy.state_dict())

        # Reinitialize the policy and load from memory
        self.policy = ActorCritic(self.num_state, self.num_action, self.action_std).to(self.device)
        self.policy.load_state_dict(initial_policy_params)

        # Freeze specified layers
        for w in range(self.freeze_num):
            self.policy.actor[2 * w].weight.requires_grad = False
            self.policy.actor[2 * w].bias.requires_grad = False
            self.policy.critic[2 * w].weight.requires_grad = False
            self.policy.critic[2 * w].bias.requires_grad = False

        # Update optimizer to only train unfrozen parameters
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=self.learning_rate,
            betas=self.betas
        )

        # Update old policy
        self.policy_old = ActorCritic(self.num_state, self.num_action, self.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
