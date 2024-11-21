# Class for the replay buffer

from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Initialize the replay buffer with a maximum size.
        
        Args:
            max_size (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque(maxlen=max_size)  # circulating deque to store experiences

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.
        
        Args:
            state (array-like): The current state of the environment.
            action (int): The action executed by the agent.
            reward (float): The reward received after executing the action.
            next_state (array-like): The state of the environment after excuting the action.
            done (bool): Whether the current episode has ended.
        """
        # Add the experience tuple to the buffer (FIFO)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a minibatch of experiences from the replay buffer.
        
        Args:
            batch_size (int): The number of experiences to sample.
        
        Returns:
            tuple: A tuple containing arrays of states, actions, rewards, next_states, and dones.
        """
        import random
        minibatch = random.sample(self.buffer, batch_size)  # Randomly sample experiences
        # Split the minibatch into separate arrays
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        """
        Get the current size of the replay buffer.
        
        Returns:
            int: The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)  # Return the current buffer size
