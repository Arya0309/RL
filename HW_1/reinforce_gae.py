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
writer = SummaryWriter(os.path.join(current_dir, "logs", "LunarLander_gae"))


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
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
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
        hidden_1 = F.relu(self.shared_layer(state))
        hidden_2 = F.relu(self.hidden_layer(hidden_1))
        action_logits = self.actor_layer(hidden_2)
        action_prob = F.softmax(action_logits, dim=1)
        state_value = self.critic_layer(hidden_2)
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
        # If the state is not a torch tensor, convert it from a numpy array, add a batch dimension.
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().unsqueeze(0)
        # Use the forward method to obtain action probabilities and the state value for the current state.
        action_prob, state_value = self.forward(state)
        # Create a Categorical distribution based on the action probabilities.
        m = Categorical(action_prob)
        # Sample an action from the distribution.
        action = m.sample()
        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.99, lambda_=0.95):
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

        ########## YOUR CODE HERE (8-15 lines) ##########
        # Create an instance of GAE using the specified gamma and lambda_ values.
        gae = GAE(gamma, lambda_)
        # Compute the advantages by passing the list of rewards, the predicted values (extracted from saved actions), and the done flags.
        advantages = gae(
            self.rewards, [value for _, value in saved_actions], self.dones
        )
        # For each saved action and its corresponding computed advantage...
        for (log_prob, value), gae in zip(saved_actions, advantages):
            # Multiply the negative log probability by the advantage to form the policy loss component.
            policy_losses.append(gae * -log_prob)
            # Compute the value loss as the MSE between the current value estimate and the sum of the value and the advantage (target).
            value_losses.append(nn.MSELoss()(value, torch.tensor(gae) + value))
        # Sum all the policy losses and value losses to obtain the total loss.
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.dones[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps=None):
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
        # Initialize next_value as 0 for bootstrapping the advantage computation.
        next_value = 0
        # Initialize the running GAE accumulator as 0.
        gae = 0
        # Create an empty list to store computed advantages for each timestep.
        advantages = []
        # Loop over rewards, values, and done flags in reverse order.
        for reward, value, done in zip(
            reversed(rewards), reversed(values), reversed(done)
        ):
            # Calculate the temporal-difference error (delta) using the reward, discounted next value (if not done), and current value.
            delta = reward + self.gamma * next_value * (1 - done) - value
            # Update the running GAE by incorporating delta and the discounted previous GAE (if not done).
            gae = delta + self.gamma * self.lambda_ * gae * (1 - done)
            # Set next_value to the current value for the next iteration.
            next_value = value
            # Insert the computed advantage at the beginning of the advantages list.
            advantages.insert(0, gae)
        # Convert the list of advantages into a torch tensor.
        advantages = torch.tensor(advantages)
        # Normalize the advantages by subtracting the mean and dividing by the standard deviation for stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # Return the normalized advantages.
        return advantages
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
            # Use the model to select an action given the current state.
            action = model.select_action(state)
            # Execute the action in the environment and receive the next state, reward, done flag, and any additional info.
            state, reward, done, _ = env.step(action)
            # Record the reward obtained at this step.
            model.rewards.append(reward)
            # Record the done flag for this step.
            model.dones.append(done)
            # Accumulate the episode's total reward.
            ep_reward += reward
            # Increment the step counter.
            t += 1

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
    env = gym.make("LunarLander-v2")
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f"LunarLander_{lr}.pth")
