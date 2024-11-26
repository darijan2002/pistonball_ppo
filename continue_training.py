import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from pettingzoo.butterfly import pistonball_v6
from tqdm import tqdm, trange


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
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)  # (batch, channel, height, width)
    obs = torch.tensor(obs).to(device)
    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{device=}')
    ent_coef = 0.03430934688621916
    vf_coef = 0.00010635186909030962
    clip_coef = 0.3848463970510941
    gamma = 0.9721757249192812
    batch_size = 60
    lr = 2.6575473238647402e-05
    # ent_coef = 0.1
    # vf_coef = 0.1
    # clip_coef = 0.1
    # gamma = 0.99
    # batch_size = 32
    # lr = 0.001
    stack_size = 4
    frame_size = (64, 64)
    max_cycles = 125
    total_episodes = 150

    """ ENV SETUP """
    env = pistonball_v6.parallel_env(
        render_mode="rgb_array", continuous=False, max_cycles=max_cycles
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # Directory to save models
    finetuning_dir = "finetuning"
    if not os.path.exists(finetuning_dir):
        os.makedirs(finetuning_dir)

    # Load the model if exists, otherwise start from scratch
    checkpoint_path = "checkpoint/finetuning_episode_5.299.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        agent.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint found, starting from scratch")

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros(
        (max_cycles, num_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    for episode in trange(1, total_episodes+1):
        agent.train()
        # collect an episode
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

                total_episodic_return += rb_rewards[step].cpu().numpy()

                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # Bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(
            rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(
            rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(
            rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(
            rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
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

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        mean_episodic_return = np.mean(total_episodic_return)
        print(f"Training episode {episode}")
        print(f"Episodic Return: {mean_episodic_return}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

        if mean_episodic_return >= 80:
            finetuning_path = os.path.join(
                finetuning_dir, f"pistonball_{episode}_rew_{int(mean_episodic_return)}.pth")
            torch.save(agent.state_dict(), finetuning_path)
            print(
                f"Model saved at episode {episode} with mean episodic reward: {mean_episodic_return}")

        if episode % 10 == 0:
            human_env = pistonball_v6.parallel_env(
                render_mode="human", continuous=False)
            human_env = color_reduction_v0(human_env)
            human_env = resize_v1(human_env, 64, 64)
            human_env = frame_stack_v1(human_env, stack_size=4)

            agent.eval()

            with torch.no_grad():
                # render 5 episodes out
                for episode in range(2):
                    obs, infos = human_env.reset(seed=None)
                    obs = batchify_obs(obs, device)
                    terms = [False]
                    truncs = [False]
                    while not any(terms) and not any(truncs):
                        actions, logprobs, _, values = agent.get_action_and_value(
                            obs)
                        obs, rewards, terms, truncs, infos = human_env.step(
                            unbatchify(actions, human_env))
                        obs = batchify_obs(obs, device)
                        terms = [terms[a] for a in terms]
                        truncs = [truncs[a] for a in truncs]

            human_env.close()

    # Save the model after each episode in the finetuning directory
    finetuning_path = os.path.join(
        finetuning_dir, f"pistonball_{episode}.pth")
    torch.save(agent.state_dict(), finetuning_path)
    print(
        f"Model saved at episode {episode} with total reward: {total_episodic_return}")
