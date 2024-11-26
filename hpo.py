import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
import optuna


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)
    obs = torch.tensor(obs).to(device)
    return obs


def batchify(x, device):
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 1e-6, 1e-1, log=True)
    clip_coef = trial.suggest_float("clip_coef", 0.1, 0.4)
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    batch_size = trial.suggest_int("batch_size", 16, 64, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    env = pistonball_v6.parallel_env(
        render_mode="rgb_array", continuous=False, max_cycles=125
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, 64, 64)
    env = frame_stack_v1(env, stack_size=4)
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n

    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # Episode setup
    max_cycles = 125
    total_episodes = 300

    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, 4, 64, 64)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    # Training loop
    for episode in range(total_episodes):
        with torch.no_grad():
            next_obs, info = env.reset(seed=None)
            total_episodic_return = 0

            for step in range(0, max_cycles):
                obs = batchify_obs(next_obs, device)
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env))

                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                total_episodic_return += rb_rewards[step].sum().item()

                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

        # Bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(max_cycles)):
                if t == max_cycles - 1:  # Last time step
                    delta = rb_rewards[t] - rb_values[t]
                else:
                    delta = (
                        rb_rewards[t]
                        + gamma * rb_values[t + 1] * (1 - rb_terms[t + 1])
                        - rb_values[t]
                    )
                rb_advantages[t] = delta + gamma * \
                    rb_advantages[t + 1] if t < max_cycles - 1 else delta
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:max_cycles], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(
            rb_logprobs[:max_cycles], start_dim=0, end_dim=1)
        b_actions = torch.flatten(
            rb_actions[:max_cycles], start_dim=0, end_dim=1)
        b_returns = torch.flatten(
            rb_returns[:max_cycles], start_dim=0, end_dim=1)
        b_values = torch.flatten(
            rb_values[:max_cycles], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(
            rb_advantages[:max_cycles], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        for repeat in range(3):
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / \
                    (advantages.std() + 1e-8)

                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Report the reward for the episode
        trial.report(total_episodic_return, episode)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return total_episodic_return


if __name__ == "__main__":
    study_name = "pistonball_hyperparam_search"  # Set the study name
    """
    storage = "sqlite:///optuna_study.db"  # Use SQLite as storage for persistence

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,  # Optional: load an existing study with the same name
    )
    # Run 10 trials for hyperparameter search
    study.optimize(objective, n_trials=10)
    """

    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///optuna/optuna_study_100.db"
    )
    print(study.best_params)