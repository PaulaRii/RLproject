# **************************************************************************************************
# **************************************** TESTING LOOP ********************************************

# Notify that testing start
print("Starting testing...")  

# Initialize testing parameters
test_episodes = 20  # number of episodes to test the agent
test_epsilon = 0.01  # minimal epsilon value for exploitation
scores = []  # for storing the scores (total rewards) for each episode

for episode in range(test_episodes):
    state, info = env.reset()  # resetting the environment at the start of each episode
    done = False
    total_reward = 0  # tracking the total reward in this episode

    while not done:
        env.render()  # rendering the environment (this is optional)
        action = epsilon_greedy_policy(state, model, test_epsilon)  # useing the learned policy
        next_state, reward, terminated, truncated, info = env.step(action)  # taking action
        total_reward += reward  # accumulating rewards
        done = terminated or truncated  # checking if the episode is over
        state = next_state  # moving to the next state

    # Append the total reward of this episode
    scores.append(total_reward)  
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Calculate and print average score
average_score = np.mean(scores)
print(f"\nAgent's average score over {test_episodes} test episodes: {average_score}")

# ***********************************************************************************************