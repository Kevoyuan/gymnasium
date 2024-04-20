import gymnasium as gym
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt

class CartPoleAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, initial_epsilon=1.0, epsilon_decay=0.99):
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor
        self.env.reset()
        # Discretize the space into bins. CartPole has 4 observations.
        # The number of bins needs careful selection to not create an overly large Q-table.
        self.bins = [
            np.linspace(-4.8, 4.8, 10),   # For CartPole observation 1
            np.linspace(-4, 4, 10),       # For CartPole observation 2
            np.linspace(-0.418, 0.418, 10),  # For CartPole observation 3
            np.linspace(-4, 4, 10)        # For CartPole observation 4
        ]
        # Create a Q-table for each combination of discretized states for each action
        self.q_table = np.random.uniform(low=-1, high=1, size=([len(b) + 1 for b in self.bins] + [self.env.action_space.n]))

    def discretize_state(self, state):
        # Discretizes the continuous state into discrete bins
        discrete_state = []
        for i, val in enumerate(state):
            print("Discretizing:", val)
            if isinstance(val, dict) or isinstance(self.bins[i], dict):
                print("Unexpected dictionary found!")
                continue
            index = np.digitize(val, self.bins[i]) - 1
            discrete_state.append(index)

        return tuple(discrete_state)



    def train(self, episodes=500):
        rewards = []
        for episode in range(episodes):
            current_state = self.discretize_state(self.env.reset())
            done = False
            total_reward = 0
            
            while not done:
                # Epsilon-greedy strategy for exploration and exploitation
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.q_table[current_state])
                    # Ensure the chosen action is within the valid range
                    action = np.clip(action, 0, self.env.action_space.n - 1)
                else:
                    action = self.env.action_space.sample()

                # st.info(self.env.step(action))
                next_state_raw, reward, done, _, info = self.env.step(action)
                next_state = self.discretize_state(next_state_raw)

                # Update Q-table using the Bellman equation
                old_value = self.q_table[current_state + (action,)]
                next_max = np.max(self.q_table[next_state])

                # Q-learning formula
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
                self.q_table[current_state + (action,)] = new_value

                current_state = next_state
                total_reward += reward
                # st.info(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            self.epsilon *= self.epsilon_decay  # Decay the exploration rate after each episode
            rewards.append(total_reward)
        st.info(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        return rewards

    def render(self, episodes):
        frames = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_start_time = time.time()  # Start time of the episode

            while not done:
                current_state = self.discretize_state(state)
                action = np.argmax(self.q_table[current_state])  # Choose best action based on Q-table
                action = np.clip(action, 0, self.env.action_space.n - 1)  # Ensure valid action

                state, reward, done, info, _ = self.env.step(action)  # Perform action
                if episode == episodes-1: 
                    # Capture the current time to show on the stopwatch
                    elapsed_time = time.time() - episode_start_time  # Stopwatch time
                    frame = self.env.render()  # Capture frame

                    # Create an annotated frame with Matplotlib
                    fig, ax = plt.subplots()
                    ax.imshow(frame)
                    annotation_text = f"Episode: {episode + 1}, Action: {action}, Elapsed Time: {elapsed_time:.2f}s, Reward: {reward}"
                    ax.text(
                        0.5,
                        0.95,
                        annotation_text,
                        color="white",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.7),
                    )
                    ax.axis("off")

                    # Convert Matplotlib figure to an image array
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    frames.append(image)  # Append the annotated frame

                    plt.close(fig)  # Close the figure to free up memory

            episode_end_time = time.time()  # Record the end time of the episode
            episode_duration = episode_end_time - episode_start_time  # Calculate the duration
            if episode == episodes-1:  # Log the duration only for the last episode
                st.info(f"Episode {episode + 1} duration: {episode_duration:.2f}s")


        self.env.close()
        return frames


