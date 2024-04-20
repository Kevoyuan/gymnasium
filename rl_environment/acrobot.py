import gymnasium as gym
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time

# acrobot.py
class AcrobotAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95):
        self.env = gym.make("Acrobot-v1", render_mode="rgb_array")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.env.reset()
        bins = 10
        # Assuming each of the 6 state variables is discretized into 'bins' number of discrete values
        self.q_table = np.zeros(
            (bins ** self.env.observation_space.shape[0], self.env.action_space.n)
        )

    def train(self, episodes=500):
        progress_bar = st.progress(0)  # Initialize the progress bar
        rewards = []  # Initialize a list to store rewards for plotting

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0  # Initialize total_reward for this episode
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, info, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward  # Accumulate reward

                if done:
                    break

            rewards.append(total_reward)  # Store the total reward for the episode
            progress = (episode + 1) / episodes  # Calculate progress percentage
            progress_bar.progress(progress)  # Update the progress bar

        self.env.close()  # Ensure the environment is closed after training
        return rewards  # Return the list of rewards for plotting

    def discretize_state(self, state, bins=10):
        if isinstance(state, tuple):
            state = state[0]
        # st.text(f"State type before conversion: {type(state)}")
        # Ensure state is a numpy array for element-wise operations
        state = np.array(state)
        # st.text(f"State type after conversion: {type(state)}")

        # Normalize the state between [0, 1]
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        normalized_state = (state - low) / (high - low)

        # Scale the normalized state to the bin range and ensure it's within bounds
        scaled_state = np.clip(np.floor(normalized_state * bins), 0, bins - 1).astype(
            int
        )

        # Convert the scaled, discrete states into a single index for the Q-table
        discrete_index = np.ravel_multi_index(scaled_state, [bins] * len(state))

        return discrete_index

    def learn(self, state, action, reward, next_state):
        # Convert states to discrete values if they are continuous

        discrete_state = self.discretize_state(state)
        # st.text(f"Discrete State: {discrete_state}")
        discrete_next_state = self.discretize_state(next_state)
        # st.text(f"Discrete Next State: {discrete_next_state}")
        # Current Q value
        current_q = self.q_table[discrete_state, action]

        # Maximum Q value for the next state
        max_future_q = np.max(self.q_table[discrete_next_state])

        # Q-learning formula
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q
        )

        # Update the Q-table with the new Q value
        self.q_table[discrete_state, action] = new_q

    def render(self, episodes):
        frames = []
        for episode in range(episodes):
            self.env.reset()
            done = False
            episode_start_time = time.time()  # Start time of the episode

            while not done:
                action = self.env.action_space.sample()  # Sample a random action
                _, _, done, _, _ = self.env.step(action)

                # Capture the current time to show on the stopwatch
                current_time = time.time()
                elapsed_time = current_time - episode_start_time  # Stopwatch time

                frame = self.env.render()  # Capture frame

                # Create an annotated frame with Matplotlib
                fig, ax = plt.subplots()
                ax.imshow(frame)
                # Annotation text for episode and stopwatch time
                annotation_text = f"Action: {action}, Elapsed Time: {elapsed_time:.2f}s"
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
            episode_duration = (
                episode_end_time - episode_start_time
            )  # Calculate the duration of the episode
            # If you want to print or log the duration of each episode, you can do it here
            st.text(f"duration: {episode_duration:.2f}s")

        self.env.close()
        return frames
