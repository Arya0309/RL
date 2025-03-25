# Spring 2025, 535514 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

# Define a tensorboard writer
current_dir = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(os.path.join(current_dir, "tb_record_1"))


class Policy(nn.Module):
    """
    Implement both policy network and the value network in one model
    - Note that here we let the actor and value networks share the first layer
    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
    - Feel free to add any member variables/functions whenever needed
    TODO:
        1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
        2. Random weight initialization of each layer
    """

    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = (
            env.action_space.n if self.discrete else env.action_space.shape[0]
        )
        self.hidden_size = 512
        self.double()

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.actor_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.critic_layer = nn.Linear(self.hidden_size, 1)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.dones = []

    def forward(self, state):
        """
        Forward pass of both policy and value networks
        - The input is the state, and the outputs are the corresponding
          action probability distirbution and the state value
        TODO:
            1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        hidden = torch.relu(self.shared_layer(state))
        action_logits = self.actor_layer(hidden)
        action_prob = torch.softmax(action_logits, dim=-1)
        state_value = self.critic_layer(hidden)
        ########## END OF YOUR CODE ##########

        return action_prob, state_value

    def select_action(self, state):
        """
        Select the action given the current state
        - The input is the state, and the output is the action to apply
        (based on the learned stochastic policy)
        TODO:
            1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.999):
        """
        Calculate the loss (= policy loss + value loss) to perform backprop later
        TODO:
            1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
            2. Calculate the policy loss using the policy gradient
            3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        gae = GAE(gamma, 0.95)
        advantages = gae(self.rewards, [sa.value for sa in saved_actions], self.dones)

        for (log_prob, value), R, advantage in zip(saved_actions, returns, advantages):
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.dones[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps=8):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps  # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########
        T = len(rewards)
        gae = [0] * T
        for t in range(T):
            gae_t = 0.0
            discount = 1.0
            # 從 t 開始往後計算最多 num_steps 步
            for k in range(t, min(t + self.num_steps, T)):
                # 注意：這裡需要處理 k+1 超出範圍的情況
                next_value = values[k + 1] if k + 1 < len(values) else 0
                delta = rewards[k] + self.gamma * next_value * (1 - done[k]) - values[k]
                gae_t += discount * delta
                discount *= self.gamma * self.lambda_
            gae[t] = gae_t
        return gae
        ########## END OF YOUR CODE ##########


def train(lr=0.01):
    """
    Train the model using SGD (via backpropagation)
    TODO (1): In each episode,
    1. run the policy till the end of the episode and keep the sampled trajectory
    2. update both the policy and the value network at the end of episode

    TODO (2): In each episode,
    1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########
        done = False
        while t < env.spec.max_episode_steps and not done:
            action = model.select_action(torch.from_numpy(state).float())
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            model.dones.append(1 if done else 0)
            ep_reward += reward
            t += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()

        model.clear_memory()
        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(
            "Episode {}\tlength: {}\treward: {}\t ewma reward: {}".format(
                i_episode, t, ep_reward, ewma_reward
            )
        )

        # Try to use Tensorboard to record the behavior of your implementation
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar("training/ep_reward", ep_reward, i_episode)
        writer.add_scalar("training/ewma_reward", ewma_reward, i_episode)
        writer.add_scalar("training/loss", loss, i_episode)
        ########## END OF YOUR CODE ##########
        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > 120:
            if not os.path.isdir(os.path.join(current_dir, "preTrained")):
                os.mkdir(os.path.join(current_dir, "preTrained"))
            torch.save(
                model.state_dict(),
                os.path.join(current_dir, "preTrained", "LunarLander_{}.pth").format(
                    lr
                ),
            )
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(ewma_reward, t)
            )
            break


def test(name, n_episodes=10):
    """
    Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(
        torch.load(
            os.path.join(current_dir, "preTrained", name),
        )
    )

    render = False
    max_episode_len = 10000

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len + 1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print("Episode {}\tReward: {}".format(i_episode, running_reward))
    env.close()


if __name__ == "__main__":
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.01
    env = gym.make("LunarLander-v2")
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f"LunarLander_{lr}.pth")