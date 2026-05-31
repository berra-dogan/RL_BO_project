from utils import Scaler, Memory
import torch
import numpy as np
from encoder import DeepSetEncoder
from environment import Env_encoder
import torch.optim as optim
from rl_agents import PPO_Agent
from torch.distributions import MultivariateNormal
from config import PPOConfig, RLBOConfig

class RL_BO():
    def __init__(self, config: RLBOConfig, ppo_config: PPOConfig | None = None, device = "cpu"):

        self.config = config
        self.ppo_config = ppo_config or PPOConfig()
        self.scaler_EI = Scaler()

        self.device = device

    def _verify_runtime_device(self, encoder, agent):
        expected_type = torch.device(self.device).type
        encoder_type = next(encoder.parameters()).device.type
        policy_type = next(agent.policy.parameters()).device.type
        policy_old_type = next(agent.policy_old.parameters()).device.type

        assert encoder_type == expected_type, f"encoder on {encoder_type}, expected {expected_type}"
        assert policy_type == expected_type, f"policy on {policy_type}, expected {expected_type}"
        assert policy_old_type == expected_type, f"policy_old on {policy_old_type}, expected {expected_type}"

    def evaluate(
        self,
        model,
        y_max_raw,
        X_train_scaled,
        y_train_raw,
        action_min_scaled,
        action_max_scaled,
        policy_file_num,
        horizon,
        movement_budget_remaining=None,
        movement_budget_total=None,
    ):
        # Features stay in scaled space; targets stay in original space and are scaled only for GP fitting.
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.scaler_EI.fit(y_train_raw)
        input_dim = X_train_scaled.shape[1] + 1
        encoder = DeepSetEncoder(input_dim=input_dim, hidden_dim=64, output_dim=16).to(self.device)
        env = Env_encoder(
            model,
            encoder,
            y_max_raw,
            X_train_scaled,
            y_train_raw,
            self.scaler_EI,
            action_min_scaled,
            action_max_scaled,
            reward_mode=self.config.reward_mode,
            snake_path_cost_weight=self.config.snake_path_cost_weight,
            reward_params=self.config.reward_params,
            movement_budget_remaining=movement_budget_remaining,
            movement_budget_total=movement_budget_total,
            device=self.device,
        )
        encoder_optimizer = optim.Adam(
            encoder.parameters(),
            lr=self.config.encoder_learning_rate,
            betas=self.config.encoder_betas,
        )

        memory = Memory()
        agent = PPO_Agent(env.num_state, env.num_action, config=self.ppo_config, device=self.device)
        self._verify_runtime_device(encoder, agent)

        eval_scores = []
        return_sum = 0
        final_average_score = 0
        no_improvement_count = 0
       
        # Main training loop
        for e in range(1, self.config.max_episodes + 1):
            state = env.reset().to(self.device)
            # print(horizon)
            max_step = horizon
            total_reward = 0
            for k in range(max_step):
                if e <= self.config.off_policy_episodes:
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
            if e % self.config.update_episode == 0:
                if e <= self.config.off_policy_episodes:
                    agent.update_initial(memory)
                    agent.transfer_learning(policy_file_num)
                else:
                    agent.update(memory, encoder_optimizer)
                memory.clear_memory()
                score = return_sum / self.config.update_episode
                eval_scores.append(score)
                print(f'Episode {e}, Average score: {score}')
                return_sum = 0

                # Check for no improvement
                if score < 1e-5:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                # If no improvement for 15 consecutive times, terminate and use turbo acquisition
                if (
                    self.config.reward_mode != "budgeted_exploration"
                    and no_improvement_count >= self.config.no_improvement_threshold
                ):
                    print(
                        f"No improvement for {self.config.no_improvement_threshold * self.config.update_episode} episodes. Terminating learning.")
                    x_next = env.turbo_acquisition()
                    return env.project_to_remaining_budget(x_next)

                if e == self.config.max_episodes:
                    final_average_score = score

        # Choose final action based on performance
        if self.config.reward_mode != "budgeted_exploration" and final_average_score < 1e-5:
            print("Final average score is less than 1e-5. Using turbo acquisition.")
            x_next = env.turbo_acquisition()
        else:
            print("Using agent's action.")
            state = env.reset()
            x_next = agent.get_action_eval(state)
            x_next = env.to_action(x_next)

        x_next = env.project_to_remaining_budget(x_next)
        X_next = np.ravel(x_next)
        return X_next.reshape(1, -1)
