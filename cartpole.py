
# For the environment and for rendering the CartPole game
import gymnasium as gym  # OpenAI Gymnasium library
import numpy as np  # For handling arrays and numerical operations
import matplotlib.pyplot as plt  # For optional visualization
import time # For adding delay
from Replay_buffer import ReplayBuffer  # Import the replay buffer
from tensorflow.keras.models import load_model # for if there is already trained model which can be used 

# For the neural network
import tensorflow as tf  # TensorFlow library
from tensorflow.keras import Sequential  # For creating neural networks
from tensorflow.keras.layers import Dense  # For defining dense layers

# Initialize the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode='human')

# Initialize the replay buffer
replay_buffer = ReplayBuffer(max_size=1000)

# *************************************************************************************************************
# *************************************** NEURAL NETWORK ******************************************************

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

# ************************************************************************************************************
# ************************************* EPSILON-GREEDY POLICY *************************************************

epsilon_start = 1.0  # starting value for epsilon
epsilon_end = 0.01   # terminating value for epsilon
epsilon_decay = 0.001  # decaying value for epsilon
epsilon = epsilon_start  # current epsilon value

# Decision-making mechanism, explorating/exploiting
def epsilon_greedy_policy(state, q_network, epsilon):
    if np.random.rand() < epsilon:  # random action (exploration)
        return env.action_space.sample()
    else: # best known action (exploitation)
        q_values = q_network.predict(state[np.newaxis])  # prediction from the neural network 
        return np.argmax(q_values[0])  # selecting the action with the highest Q-value

# Reset the environment to its initial state
state, info = env.reset()

# ************************************************************************************************************
# *********************************** TARINING LOOP starts here **********************************************

# Notify that training starts
print("Starting training...")  

total_rewards = []  # list for storing total rewards of each episode
episode_reward = 0  # for keeping track of rewards in a single episode

# Run random actions in the environment
for _ in range(1000):  # Run the loop for a maximum of 50 iterations
    env.render()  # rendering the game environment
    action = epsilon_greedy_policy(state, model, epsilon) # choosing the action using the Epsilon-Greedy logic
    next_state, reward, terminated, truncated, info = env.step(action)  # executing the action

    # next_state: the new state after the action
    # reward: the reward received for taking the action
    # terminated: indicates whether the episode has ended
    # truncated: indicates whether the episode ended artificially
    # info: diagnostic information, ignored here

    # Check if the episode is over
    done = terminated or truncated

    # print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}") # test printing, REMOVE LATER!

    # Save the interaction to the replay buffer
    replay_buffer.add(state, action, reward, next_state, done)

    # Accumulate reward for this episode
    episode_reward += reward  

# **************************************** Model training **************************************************

    # Train the model if there are enough samples in the replay buffer
    if replay_buffer.size() >= 128:  # NEED TO BE SURE THAT there are enough samples for a meaningful batch!
        batch = replay_buffer.sample(batch_size=128) # 64 or 128
        states, actions, rewards, next_states, dones = batch

        # Predict Q-values for current and next states
        q_values = model.predict(states) # action's value based on the current state
        next_q_values = model.predict(next_states) # estimate of the Q-value for the next state

        # Compute target Q-values (Bellman's equation!)
        target_q_values = q_values.copy()

        for i in range(len(rewards)):
            # The Bellman's equation for calculating the Q-value
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * 0.99 * np.max(next_q_values[i]) 
            
            # dones[i]: preventing for counting the future Q-value estimates if the episode is terminated
            # 0.99 : gamma-value, the discount factor 
            # np.max(next_q_values[i]): biggest estimated Q-value for the next state
        
        # Train the model by using the result from the seitäBellman's equation
        model.train_on_batch(states, target_q_values) # the estimated Q-values are used here forn training the model

    # Decay the epsilon so the agent explores less over time and relies more to the best known actions
    if epsilon > epsilon_end:
        epsilon -= epsilon_decay

    # This exponential decrease for epsilon could be used instead of the decaying method above
    # epsilon = max(epsilon_end, epsilon * decay_rate)

    # Update the current state
    if done:
        state, info = env.reset()  # resetting the environment if the episode is over
    else:
        state = next_state

    # If episode ends, record total reward 
    if terminated or truncated:
        total_rewards.append(episode_reward)
        episode_reward = 0

# Notify that training ends
print("Training completed. Saving the model...")  

# Save the trained model into a file
model.save('cartpole_model.keras')
print("Model saved to cartpole_model.h5")

# ************************************** TRAINING LOOP ends here ************************************
# ***************************************************************************************************

# Display a sample from the replay buffer
if replay_buffer.size() >= 1:  # ensuring there's at least one experience in the buffer
    batch = replay_buffer.sample(batch_size=1)
    print("Sampled batch:", batch)

# Wait before closing the windown
time.sleep(8) 

# Calculate average reward over training
if total_rewards:
    print(f"Average training reward over {len(total_rewards)} episodes: {np.mean(total_rewards)}")

# Close the environment
env.close()
