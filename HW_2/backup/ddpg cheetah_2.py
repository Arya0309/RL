# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG (with GPU support and efficient tensor conversions)

import copy
import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

torch.set_default_dtype(torch.float32)

current_dir = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(os.path.join(current_dir, "logs_2/tb_3"))


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple(
    "Transition", ("state", "action", "mask", "next_state", "reward")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        num_outputs = action_space.shape[0]
        self.fc1 = nn.Linear(num_inputs, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, num_outputs)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super().__init__()
        num_outputs = action_space.shape[0]
        self.fc1 = nn.Linear(num_inputs + num_outputs, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, action):
        x = F.relu(self.ln1(self.fc1(torch.cat([x, action], 1))))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class DDPG:
    def __init__(
        self,
        num_inputs,
        action_space,
        gamma=0.995,
        tau=0.0005,
        hidden_size=128,
        lr_a=1e-4,
        lr_c=3e-4,
    ):
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        # actor/critic + targets
        self.actor = Actor(hidden_size, num_inputs, action_space).to(device)
        self.actor_target = Actor(hidden_size, num_inputs, action_space).to(device)
        self.critic = Critic(hidden_size, num_inputs, action_space).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        # initial hard update
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(state)
        mu = mu.cpu().numpy().flatten()
        if action_noise is not None:
            mu += action_noise.noise()
        mu = np.clip(mu, self.action_space.low, self.action_space.high)
        return torch.from_numpy(mu).float().to(device).unsqueeze(0)

    def update_parameters(self, batch):
        # concatenate and move to device
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).unsqueeze(1).to(device)
        mask_batch = torch.cat(batch.mask).unsqueeze(1).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # Critic update
        q = self.critic(state_batch, action_batch)
        next_a = self.actor_target(next_state_batch)

        # Compute target Q value
        target_q = self.critic_target(next_state_batch, next_a)
        target_q = reward_batch + (mask_batch * self.gamma * target_q).detach()

        critic_loss = F.mse_loss(q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft updates
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        timestamp = time.strftime("%m%d%Y_%H%M%S", time.localtime())
        os.makedirs("preTrained", exist_ok=True)
        if actor_path is None:
            actor_path = f"preTrained/ddpg_actor_{env_name}_{timestamp}{suffix}"
        if critic_path is None:
            critic_path = f"preTrained/ddpg_critic_{env_name}_{timestamp}{suffix}"
        print(f"Saving models to {actor_path} and {critic_path}")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print(f"Loading models from {actor_path} and {critic_path}")
        if actor_path:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path:
            self.critic.load_state_dict(torch.load(critic_path))


def train():
    random_seed = 10
    env = gym.make("HalfCheetah")
    env.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    env.action_space.seed(random_seed)
    env.observation_space.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_episodes = 500
    gamma = 0.99
    tau = 0.001
    hidden_size = 400
    noise_scale = 0.3
    final_noise_scale = 0.05
    decay_episodes = 300
    replay_size = 100_000
    batch_size = 256
    updates_per_step = 1
    print_freq = 1

    agent = DDPG(
        env.observation_space.shape[0],
        env.action_space,
        gamma=gamma,
        tau=tau,
        hidden_size=hidden_size,
    )
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    ewma_reward = 0.0
    ewma_reward_history = []
    updates = 0

    for i_episode in range(num_episodes):
        if i_episode < decay_episodes:
            ounoise.scale = noise_scale
        else:
            # linear decay to final_noise_scale
            ounoise.scale = (
                noise_scale
                - (i_episode - decay_episodes) / (num_episodes - decay_episodes)
                * (noise_scale - final_noise_scale)
            )
        ounoise.reset()

        # reset env & convert
        state_np = env.reset(seed=random_seed)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        episode_reward = 0.0

        while True:
            action = agent.select_action(state, action_noise=ounoise)

            next_s_np, r, done, _ = env.step(action.cpu().numpy()[0])
            episode_reward += r

            next_state = torch.from_numpy(next_s_np).float().unsqueeze(0).to(device)
            reward = torch.tensor([r], dtype=torch.float32, device=device)
            mask = torch.tensor(
                [0.0 if done else 1.0], dtype=torch.float32, device=device
            )

            memory.push(state, action, mask, next_state, reward)
            state = next_state

            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    critic_loss, actor_loss = agent.update_parameters(batch)
                    updates += 1
                    writer.add_scalar("actor_loss", actor_loss, updates)
                    writer.add_scalar("critic_loss", critic_loss, updates)
                    writer.add_scalar("reward", episode_reward, updates)
                    writer.add_scalar("ewma_reward", ewma_reward, updates)

            if done:
                break

        # evaluation
        if i_episode % print_freq == 0:
            test_reward = 0.0
            s_np = env.reset(seed=random_seed)
            s = torch.from_numpy(s_np).float().unsqueeze(0).to(device)
            done = False
            t = 0
            while not done:
                a = agent.select_action(s).cpu().numpy()[0]
                s2_np, r2, done, _ = env.step(a)
                test_reward += r2
                s = torch.from_numpy(s2_np).float().unsqueeze(0).to(device)
                t += 1

            ewma_reward = 0.05 * test_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)
            print(
                f"Episode: {i_episode}, length: {t}, reward: {test_reward:.2f}, ewma: {ewma_reward:.2f}"
            )

        if ewma_reward > 5000:
            print(f"Early stopping at episode {i_episode}")
            break

    agent.save_model("HalfCheetah", suffix=".pth")


if __name__ == "__main__":
    train()