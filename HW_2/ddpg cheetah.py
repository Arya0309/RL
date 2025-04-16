# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG

import d4rl
import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn

# import wandb
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float32)
current_dir = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(os.path.join(current_dir, "logs/tb_1"))

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple(
    "Transition", ("state", "action", "mask", "next_state", "reward")
)


class ReplayMemory(object):
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
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
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
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            bound = 1.0 / np.sqrt(fan_in)
            return (-bound, bound)

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            bound = 1.0 / np.sqrt(fan_in)
            return (-bound, bound)

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, inputs, actions):
        x = F.relu(self.fc1(inputs))
        x = torch.cat([x, actions], 1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class DDPG(object):
    def __init__(
        self,
        num_inputs,
        action_space,
        gamma=0.995,
        tau=0.0005,
        hidden_size=128,
        lr_a=1e-4,
        lr_c=1e-3,
    ):
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data  # This is still float32 from the network

        if action_noise is not None:
            # Convert mu to NumPy, add OU noise, then clamp and return as FloatTensor
            mu = mu.cpu().numpy()
            mu += action_noise.noise()
            mu = np.clip(mu, self.action_space.low, self.action_space.high)
            return torch.FloatTensor(mu)
        else:
            return mu.float()  # already float32, but just ensuring

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        # 1) Critic update
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        q = self.critic(state_batch, action_batch)
        next_a = self.actor_target(next_state_batch)
        next_q = self.critic_target(next_state_batch, next_a)
        target_q = reward_batch + mask_batch * self.gamma * next_q
        value_loss = F.mse_loss(q, target_q.detach())

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # 2) Actor update
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists("preTrained/"):
            os.makedirs("preTrained/")

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(
                env_name, timestamp, suffix
            )
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(
                env_name, timestamp, suffix
            )
        print("Saving models to {} and {}".format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print("Loading models from {} and {}".format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

def train(env_name="Pendulum-v0"):
    num_episodes = 200
    gamma = 0.995
    tau = 0.002
    hidden_size = 128
    noise_scale = 0.3
    batch_size = 128
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    agent = DDPG(
        env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size
    )

    env = gym.make(env_name)
    env.reset(seed=random_seed)
    torch.manual_seed(random_seed)

    dataset = d4rl.get_dataset(env)

    replay_size = len(dataset['observations'])
    memory = ReplayMemory(replay_size)

    num_data = len(dataset['observations']).shape[0]
    for i in range(num_data):
        s = torch.FloatTensor([dataset['observations'][i]])
        a = torch.FloatTensor([dataset['actions'][i]])
        done_flag = dataset['terminals'][i]
        mask = torch.FloatTensor([0.0 if done_flag == 1 else 1.0])
        s_next = torch.FloatTensor([dataset['next_observations'][i]])
        r = torch.FloatTensor([dataset['rewards'][i]])
        memory.push(s, a, mask, s_next, r)
    
    print("Replay memory filled with {} transitions.".format(len(memory)))

    # 初始化 agent
    agent = DDPG(env.observation_space.shape[0], env.action_space)
    num_updates = 10000  # 可依需求設定訓練更新次數
    batch_size = 128
    updates = 0

    for update in range(num_updates):
        batch = Transition(*zip(*memory.sample(batch_size)))
        critic_loss, actor_loss = agent.update_parameters(batch)
        updates += 1
        
        # tensorboard logging
        writer.add_scalar("actor_loss", actor_loss, updates)
        writer.add_scalar("critic_loss", critic_loss, updates)
        
        if update % 1000 == 0:
            print(f"Update {update} | Actor loss: {actor_loss:.4f} | Critic loss: {critic_loss:.4f}")

    agent.save_model(env_name, suffix="offline")

    # 若需要評估，還可以開啟線上互動測試（但此處訓練為純離線模式）
    state = env.reset()
    test_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor([state])
        action = agent.select_action(state_tensor)
        state, reward, done, _ = env.step(action.numpy()[0])
        test_reward += reward
    print("Test reward:", test_reward)


if __name__ == "__main__":
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    train("halfcheetah-medium-v0")
