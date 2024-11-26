import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, resize_v1, frame_stack_v1
import os 

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

def load_model(model_path, agent, device):
    """Load a specific model from a given path."""
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        exit(1)
    return agent

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

def evaluate_agent(agent, env, num_episodes, max_cycles, device):
    """Evaluate the agent by running it on the environment for a few episodes."""
    agent.eval()  # Set the agent to evaluation mode
    total_rewards = []

    for episode in range(num_episodes):
        next_obs, _ = env.reset(seed=None)
        total_episodic_return = 0

        for step in range(max_cycles):
            obs = batchify_obs(next_obs, device)

            # Get action for all agents
            actions, _, _, _ = agent.get_action_and_value(obs)

            # Convert actions tensor to a dictionary where the key is agent name
            actions_dict = {agent_name: action.item() for agent_name, action in zip(env.possible_agents, actions)}

            # Pass the actions to the environment
            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
            total_episodic_return += sum(rewards.values())  # Sum rewards for all agents

            # Render the environment at each step
            env.render()

            # Check if any agent has finished the episode (done or truncated)
            if any(terms.values()) or any(truncs.values()):
                break

        total_rewards.append(total_episodic_return)
        print(f"Episode {episode + 1}: Total reward = {total_episodic_return}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    # Environment setup
    env = pistonball_v6.parallel_env(
        render_mode="human", continuous=False, max_cycles=125
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, 64, 64)
    env = frame_stack_v1(env, stack_size=4)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the agent
    agent = Agent(num_actions=num_actions).to(device)

    # Load the specific model from the given path
    model_path = "finetuning/pistonball_220.pth"
    model_path = "finetuning/pistonball_430.pth"
    model_path = "finetuning/pistonball_440.pth"
    model_path = "finetuning/pistonball_480.pth"
    model_path = "finetuning/pistonball_500.pth"
    model_path = "finetuning/pistonball_360.pth" #
    model_path = "finetuning/pistonball_350.pth" #
    model_path = "finetuning/pistonball_390.pth" #
    model_path = "finetuning/pistonball_470.pth" #
    model_path = "finetuning/pistonball_340.pth" #
    model_path = "finetuning/pistonball_330.pth" #
    model_path = "finetuning/pistonball_210.pth" #
    model_path = "finetuning/pistonball_240.pth" ##
    model_path = "finetuning/pistonball_250.pth" ##
    agent = load_model(model_path, agent, device)

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
            step = 0
            total_episodic_return = 0

            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs)
                obs, rewards, terms, truncs, infos = human_env.step(
                    unbatchify(actions, human_env))

                total_episodic_return += batchify(rewards, 'cpu').numpy()
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
                step += 1


            mean_episodic_return = np.mean(total_episodic_return)
            print(f"Training episode {episode}")
            print(f"Episodic Return: {mean_episodic_return}")
            print(f"Episode Length: {step}")

    human_env.close()
    # Evaluate the agent