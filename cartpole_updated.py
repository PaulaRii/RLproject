# For the environment and for rendering the CartPole game
import gymnasium as gym  # OpenAI Gymnasium library
import numpy as np  # For handling arrays and numerical operations
import matplotlib.pyplot as plt  # For optional visualization
import time  # For adding delay
from Replay_buffer import ReplayBuffer  # Import the replay buffer
from tensorflow.keras.models import load_model  # for if there is already trained model which can be used

# For the neural network
import tensorflow as tf  # TensorFlow library
from tensorflow.keras import Sequential  # For creating neural networks
from tensorflow.keras.layers import Dense  # For defining dense layers

# Initialize the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode=None) # rendering off for the training

# Initialize the replay buffer
replay_buffer = ReplayBuffer(max_size=7000)  # Increased buffer size for storing more experiences

# *************************************************************************************************************
# *************************************** NEURAL NETWORK ******************************************************

# Create the neural network for the CartPole environment
model = Sequential([
    Dense(24, activation='relu', input_shape=(4,)),  # input layer with 4 inputs (state vector size is 4)
    Dense(24, activation='relu'),  # hidden layer with 24 units and ReLU activation
    Dense(2, activation='linear')  # output layer with 2 units (actions: left/right)
])

# Compile the model with Adam optimizer and mean squared error (MSE) as the loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')

# Print the model's structure
model.summary()

# ************************************************************************************************************
# ************************************* EPSILON-GREEDY POLICY *************************************************

epsilon_start = 1.0  # starting value for epsilon
epsilon_end = 0.01  # terminating value for epsilon
epsilon_decay = 0.001  # decaying value for epsilon
epsilon = epsilon_start  # current epsilon value

# Decision-making mechanism, exploring/exploiting
def epsilon_greedy_policy(state, q_network, epsilon):
    if np.random.rand() < epsilon:  # random action (exploration)
        return env.action_space.sample()
    else:  # best known action (exploitation)
        q_values = q_network.predict(state[np.newaxis])  # prediction from the neural network
        return np.argmax(q_values[0])  # selecting the action with the highest Q-value

# ************************************************************************************************************
# *********************************** TRAINING LOOP starts here **********************************************

# Notify that training starts
print("Starting training...")

total_rewards = []  # list for storing total rewards of each episode
gamma = 0.99  # defining the discount factor (changing this value affects how much the agent values future rewards)
train_every = 10  # training the model every N steps to improve efficiency

# Run the training loop for a fixed number of steps
for step in range(6000):  # maximum number of steps in the training loop
    # env.render()  # rendering the game environment (maybe just when testing the model?)
    
    # Reset the environment at the start of each episode
    if step == 0 or done:
        state, info = env.reset()
        episode_reward = 0  # for keeping track of rewards in a single episode
        steps = 0  # tracking the number of steps in the current episode
        done = False

    # Take action using epsilon-greedy policy
    action = epsilon_greedy_policy(state, model, epsilon)
    next_state, reward, terminated, truncated, info = env.step(action)  # executing the action
    done = terminated or truncated  # check if the episode is over

    # Save the interaction to the replay buffer
    replay_buffer.add(state, action, reward, next_state, done)

    # Accumulate reward and count steps for this episode
    episode_reward += reward
    steps += 1

    # Update state
    state = next_state if not done else None  # if not done, the loop starts over for the next episode

    # Train the model if enough steps have passed and buffer has sufficient samples
    if step % train_every == 0 and replay_buffer.size() >= 128:  # model training happens only every `train_every` steps
        batch = replay_buffer.sample(batch_size=128)
        states, actions, rewards, next_states, dones = batch

        # Predict Q-values for the current and next states
        q_values = model.predict(states)  # action's value based on the states taken from the saved batch
        next_q_values = model.predict(next_states)  # estimate of the Q-value for the next states

        # Compute target Q-values
        target_q_values = q_values.copy()  # making a copy of the predicted Q-values based on the current states

        for i in range(len(rewards)):
            # !!! THE BELLMAN'S EQUATION for calculating the Q-value !!!
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * gamma * np.max(next_q_values[i])

        # Train the model by using the result from Bellman's equation
        model.train_on_batch(states, target_q_values)  # The estimated Q-values are used here for training the model

    # Decay the epsilon so the agent explores less over time and relies more on the best-known actions
    if epsilon > epsilon_end:
        epsilon -= epsilon_decay

    # Record the total reward if the episode is done
    if done:
        total_rewards.append(episode_reward)

# ************************************** TRAINING LOOP ends here ************************************

# Notify that training ends
print("Training completed. Saving the model...")

# Save the trained model into a file
model.save('cartpole_model.keras')
print("Model saved to cartpole_model.keras")

# Display a sample from the replay buffer
if replay_buffer.size() >= 1:  # ensuring there's at least one experience in the buffer
    batch = replay_buffer.sample(batch_size=1)
    print("Sampled batch:", batch)

# Calculate and print average reward
if total_rewards:
    print(f"Average training reward over {len(total_rewards)} episodes: {np.mean(total_rewards)}")

# Wait before closing the window
time.sleep(8)

# Close the environment
env.close()
