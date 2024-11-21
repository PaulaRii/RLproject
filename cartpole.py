
# For the environment and for rendering the CartPole game
import gymnasium as gym  # OpenAI Gymnasium library
import numpy as np  # For handling arrays and numerical operations
import matplotlib.pyplot as plt  # For optional visualization
import time # For adding delay
from replay_buffer import ReplayBuffer  # Import the replay buffer

# For the neural network
import tensorflow as tf  # TensorFlow library
from tensorflow.keras import Sequential  # For creating neural networks
from tensorflow.keras.layers import Dense  # For defining dense layers

# Initialize the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode='human')

# Initialize the replay buffer
replay_buffer = ReplayBuffer(max_size=100)

# ****************************************************************************************************************
# Create the neural network for the CartPole environment
model = Sequential([
    Dense(24, activation='relu', input_shape=(4,)),  # input layer with 4 inputs (state vector size is 4)
    Dense(24, activation='relu'),  # hidden layer with 24 units and ReLU activation
    Dense(2, activation='linear')  # output layer with 2 units (actions: left/right)
])

# Compile the model with Adam optimizer and mean squared error (MSE) as the loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Print the model's structure
model.summary()

# ****************************************************************************************************************

# Reset the environment to its initial state
state, info = env.reset()

# Run random actions in the environment
for _ in range(50):  # Run the loop for a maximum of 50 iterations
    action = env.action_space.sample()  # choose a random action (e.g. move left or right)
    next_state, reward, terminated, truncated, info = env.step(action)  # executing the action

    # "next_state": The new state after the action
    # "reward": The reward received for taking the action
    # "terminated": Indicates whether the episode has ended
    # "truncated": Indicates whether the episode ended artificially
    # "info": Diagnostic information, ignored here

    # Check if the episode is over
    done = terminated or truncated

    print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}") # test printing, REMOVE LATER!

    # Save the interaction to the replay buffer
    replay_buffer.add(state, action, reward, next_state, done)

    # Update the current state
    if done:
        state, info = env.reset()  # Reset the environment if the episode is over
    else:
        state = next_state

# Display a sample from the replay buffer
if replay_buffer.size() >= 1:  # ensuring there's at least one experience in the buffer
    batch = replay_buffer.sample(batch_size=1)
    print("Sampled batch:", batch)

# Wait before closing the windown
time.sleep(5) 

# Close the environment
env.close()
