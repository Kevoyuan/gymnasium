import streamlit as st
from rl_environment.cart_pole import CartPoleAgent
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Ensure the temp directory exists to save GIFs
if not os.path.exists('temp'):
    os.makedirs('temp')

def create_gif(frames, filepath="animation.gif", total_duration_s=5):
    """Creates a GIF from a list of image frames."""
    if not frames:
        raise ValueError("No frames provided for the GIF.")

    total_duration_ms = total_duration_s * 100  # Convert seconds to milliseconds
    frame_duration = 5 # Floor division to get an integer result

    # Ensure a minimum frame duration for compatibility with all viewers
    min_frame_duration = 20  # This is the approximate minimum most viewers support
    frame_duration = max(frame_duration, min_frame_duration)

    images = [Image.fromarray(frame) for frame in frames]

    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0,
    )

def configure_agent():
    """Creates a sidebar for configuring the RL agent."""
    st.sidebar.header("Configuration")
    lr = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    discount = st.sidebar.slider("Discount Factor", min_value=0.1, max_value=1.0, value=0.95, step=0.01)
    eps = st.sidebar.slider("Episodes", min_value=1, max_value=5000, value=1000)
    return lr, discount, eps

def train_agent(agent, episodes):
    """Trains the agent and plots the learning progress."""
    with st.spinner("Training..."):
        rewards = agent.train(episodes=episodes)
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Progress")
    st.pyplot(fig)

def animate_agent(agent, episodes, total_duration_s=5):
    """Generates and displays animations for the agent's performance."""
    frames = agent.render(episodes=episodes)
    if not frames:
        raise ValueError("Rendering did not produce any frames.")
    gif_path = "temp/cartpole_animation.gif"
    create_gif(frames, filepath=gif_path, total_duration_s=total_duration_s)
    st.image(gif_path, caption="Acrobot Animation")
    
    
# Application start
st.title("CartPole Reinforcement Learning")

# Initialize agent with configuration
learning_rate, discount_factor, episodes = configure_agent()
agent = CartPoleAgent(learning_rate=learning_rate, discount_factor=discount_factor)

if st.sidebar.button("Train Agent"):
    train_agent(agent, episodes)

if st.sidebar.button("Animate!"):
    animate_agent(agent, episodes)  # Animate a single episode for demonstration

# Streamlit requires a rerun to reflect changes on the UI after the first complete run
st.sidebar.text("Press 'Train Agent' to train and then 'Animate!' to see the agent in action.")
