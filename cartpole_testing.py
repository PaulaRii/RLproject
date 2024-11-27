# Code for testing the trained model

# Import required libraries
import gymnasium as gym
import numpy as np
import tensorflow as tf  # TensorFlow library for loading the model

# Initialize the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode='human')

# Load the trained model
try:
    model = tf.keras.models.load_model('cartpole_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# ******************************* Epsilon-Greedy Policy ****************************************

# Decision-making mechanism: Epsilon-Greedy Policy
def epsilon_greedy_policy(state, q_network, epsilon):
    """
    Function to choose an action using epsilon-greedy policy.
    Parameters:
        state: current state of the environment
        q_network: neural network (Q-network) to predict Q-values
        epsilon: probability for exploration (random action)
    Returns:
        action: selected action (int)
    """
    if np.random.rand() < epsilon:  # random action (exploration)
        return env.action_space.sample() # choosing a random action
    else:  # best known action (exploitation)
        q_values = q_network.predict(state[np.newaxis])  # prediction from the neural network
        return np.argmax(q_values[0])  # selecting the action with the highest Q-value

# **********************************************************************************************
# *********************************** TESTING LOOP *********************************************

# Notify that testing start
print("Starting testing...")  

# Initialize testing parameters
test_episodes = 20  # number of episodes to test the agent
test_epsilon = 0.01  # minimal epsilon value for exploitation (almost no exploration)
scores = []  # for storing the scores (total rewards) for each episode

# Test the agent for multiple episodes
for episode in range(test_episodes):
    state, info = env.reset()  # resetting the environment at the start of each episode
    done = False
    total_reward = 0  # tracking the total reward in this episode

    while not done:
        env.render()  # rendering the environment (this is optional)
        action = epsilon_greedy_policy(state, model, test_epsilon)  # useing the learned policy!
        next_state, reward, terminated, truncated, info = env.step(action)  # taking action
        total_reward += reward  # accumulating rewards
        done = terminated or truncated  # checking if the episode is over
        state = next_state  # moving to the next state

    # Adding together the total reward of this episode
    scores.append(total_reward)  
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# ***********************************************************************************************

# Calculate and print average score
average_score = np.mean(scores)
print(f"\nFinal results over {test_episodes} episodes: Min = {np.min(scores)}, Max = {np.max(scores)}, Avg = {average_score}")

# Close the environment
env.close()