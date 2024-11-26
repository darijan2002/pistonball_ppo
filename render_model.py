import numpy as np
import torch
import torch.nn as nn  # <-- Add this line
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

def load_latest_model(model_dir, agent, device):
    """Load the latest model file from a directory."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if model_files:
        latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = os.path.join(model_dir, latest_model)
        agent.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No model found in the directory.")
    return agent

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)  # (batch, channel, height, width)
    obs = torch.tensor(obs).to(device)
    return obs

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

    # Load the latest model from the directory
    model_dir = "/home/ivan/Desktop/PISTONBALL_PAK"  # Correct path to the saved models directory
    agent = load_latest_model(model_dir, agent, device)

    # Evaluate the agent
    evaluate_agent(agent, env, num_episodes=5, max_cycles=125, device=device)
