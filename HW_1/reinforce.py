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
writer = SummaryWriter("./tb_record_1")


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
        self.hidden_size = 128
        self.double()

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.actor_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.critic_layer = nn.Linear(self.hidden_size, 1)

        # Define a function for Xavier uniform initialization on linear layers.
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        Forward pass of both policy and value networks
        - The input is the state, and the outputs are the corresponding
          action probability distirbution and the state value
        TODO:
            1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Convert state to torch tensor if it's a numpy array.
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        # Pass state through the shared layer and apply ReLU activation.
        hidden = torch.relu(self.shared_layer(state))
        # Pass the hidden representation through the actor layer to get action logits.
        action_logits = self.actor_layer(hidden)
        # Apply softmax on the logits to obtain action probabilities.
        action_prob = torch.softmax(action_logits, dim=-1)
        # Pass the hidden representation through the critic layer to get the state value.
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
        # Use the forward method to get action probabilities and state value.
        action_prob, state_value = self.forward(state)
        # Create a Categorical distribution from the action probabilities.
        m = Categorical(action_prob)
        # Sample an action from the distribution.
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
        # Loop over rewards in reverse order to calculate discounted returns.
        for r in self.rewards[::-1]:
            # Update the cumulative return R as current reward plus gamma times previous return.
            R = r + gamma * R
            # Insert the computed return at the beginning of the returns list.
            returns.insert(0, R)
        # Convert the list of returns to a torch tensor.
        returns = torch.tensor(returns)
        # Normalize the returns by subtracting mean and dividing by standard deviation.
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # Iterate over saved actions paired with normalized returns.
        for (log_prob, value), R in zip(saved_actions, returns):
            # Compute the policy loss as negative log probability multiplied by return.
            policy_losses.append(-log_prob * R)
            # Compute the value loss using smooth L1 loss between predicted value and return.
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        # Sum the policy losses and value losses to get the total loss.
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps):
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
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########
        # Initialize the done flag.
        done = False
        # Loop while not done and within maximum allowed steps.
        while t < env.spec.max_episode_steps and not done:
            # Get an action from the model (ensure state is a torch tensor).
            action = model.select_action(torch.from_numpy(state).float())
            # Execute the action in the environment; receive new state, reward, and done flag.
            state, reward, done, _ = env.step(action)
            # Append the reward to the model's rewards list.
            model.rewards.append(reward)
            # Accumulate the total reward for this episode.
            ep_reward += reward
            # Increment the timestep counter.
            t += 1
        # Zero the gradients before backpropagation.
        optimizer.zero_grad()
        # Compute the loss using the stored rewards and actions.
        loss = model.calculate_loss()
        # Backpropagate the loss.
        loss.backward()
        # Update the model parameters.
        optimizer.step()
        # Clear stored rewards and actions for the next episode.
        model.clear_memory()
        ########## END OF YOUR CODE ##########

        # Try to use Tensorboard to record the behavior of your implementation
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar("training/ep_reward", ep_reward, i_episode)
        writer.add_scalar("training/ewma_reward", ewma_reward, i_episode)
        writer.add_scalar("training/loss", loss, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), "./preTrained/CartPole_{}.pth".format(lr))
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

    model.load_state_dict(torch.load("./preTrained/{}".format(name)))

    render = True
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
    env = gym.make("CartPole-v0")
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f"CartPole_{lr}.pth")
