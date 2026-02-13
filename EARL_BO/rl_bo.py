from utils import Scaler, Memory
import torch
import numpy as np
from encoder import DeepSetEncoder
from environment import Env_encoder
import torch.optim as optim
from rl_agents import PPO_Agent
from torch.distributions import MultivariateNormal

class RL_BO():
    def __init__(self, device = "cpu"):
        self.gamma = 0.95
        self.max_episodes = 100
        self.update_episode = 10
        self.scaler_EI = Scaler()
        self.off_policy_episodes = 100

        self.device = device

    def evaluate(self, model, y_max_org, X_train_org, y_train_org, action_min, action_max, policy_file_num, horizon):
        # Main method for RL-based Bayesian Optimization
        self.scaler_EI.fit(y_train_org)

        torch.manual_seed(1)
        np.random.seed(1)
        # Initialize encoder and environment
        input_dim = X_train_org.shape[1] + 1
        encoder = DeepSetEncoder(input_dim=input_dim, hidden_dim=64, output_dim=16).to(self.device)
        env = Env_encoder(model, encoder, y_max_org, X_train_org, y_train_org, self.scaler_EI, action_min, action_max)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01, betas=(0.9, 0.999))

        memory = Memory()
        agent = PPO_Agent(env.num_state, env.num_action)

        eval_scores = []
        return_sum = 0
        final_average_score = 0
        no_improvement_count = 0
        no_improvement_threshold = 15  # 15 * 50 episodes = 750 episodes

        # Main training loop
        for e in range(1, self.max_episodes + 1):
            state = env.reset().to(self.device)
            # print(horizon)
            max_step = horizon
            total_reward = 0
            for k in range(max_step):
                if e <= self.off_policy_episodes:
                    # Get action from TURBO
                    turbo_action = env.turbo_acquisition()

                    # Convert TURBO action to normalized space [-1,1]
                    normalized_action = env.reverse_action(turbo_action)
                    action_tensor = torch.FloatTensor(normalized_action).to(self.device)

                    # Calculate the log probability of this action under current policy
                    action_mean = agent.policy.actor(state)
                    cov_mat = torch.diag(agent.policy.action_var).to(self.device)
                    dist = MultivariateNormal(action_mean, cov_mat)
                    action_logprob = dist.log_prob(action_tensor)

                    # Store experience with proper log probability
                    memory.states.append(state.cpu())
                    memory.actions.append(action_tensor.cpu())
                    memory.logprobs.append(action_logprob.cpu())

                    # Use original TURBO action for environment step
                    action = turbo_action
                else:
                    # Use RL agent for on-policy learning
                    action = agent.get_action(state, memory)
                next_state, reward = env.step(action)
                total_reward += reward
                memory.rewards.append(reward)
                memory.is_terminals.append(k == max_step - 1)
                state = next_state.detach().clone()

            return_sum += total_reward

            # Update the agent periodically
            if e % self.update_episode == 0:
                if e <= self.off_policy_episodes:
                    agent.update_initial(memory)
                    agent.transfer_learning(policy_file_num)
                else:
                    agent.update(memory, encoder_optimizer)
                memory.clear_memory()
                score = return_sum / self.update_episode
                eval_scores.append(score)
                print(f'Episode {e}, Average score: {score}')
                return_sum = 0

                # Check for no improvement
                if score < 1e-5:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                # If no improvement for 15 consecutive times, terminate and use turbo acquisition
                if no_improvement_count >= no_improvement_threshold:
                    print(
                        f"No improvement for {no_improvement_threshold * self.update_episode} episodes. Terminating learning.")
                    x_next = env.turbo_acquisition()
                    return x_next.reshape(1, -1)

                if e == self.max_episodes:
                    final_average_score = score

        # Choose final action based on performance
        if final_average_score < 1e-5:
            print("Final average score is less than 1e-5. Using turbo acquisition.")
            x_next = env.turbo_acquisition()
        else:
            print("Using agent's action.")
            state = env.reset()
            x_next = agent.get_action_eval(state)
            x_next = env.to_action(x_next)

        X_next = np.ravel(x_next)
        return X_next.reshape(1, -1)

