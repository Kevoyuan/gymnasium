import streamlit as st
from rl_environment.acrobot import AcrobotAgent
import matplotlib.pyplot as plt
import numpy as np

st.title('Acrobot Reinforcement Learning')

# Sidebar for configuration
st.sidebar.header('Configuration')
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)
discount_factor = st.sidebar.slider('Discount Factor', min_value=0.8, max_value=0.99, value=0.95, step=0.01)
episodes = st.sidebar.slider('Episodes', min_value=10, max_value=1000, value=100)

if st.sidebar.button('Train Agent'):
    with st.spinner('Training...'):
        agent = AcrobotAgent(learning_rate=learning_rate, discount_factor=discount_factor)
        rewards = agent.train(episodes=episodes)  # This method now returns a list of rewards per episode

    # Plotting the rewards
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')
    st.pyplot(fig)


# Optional: Provide a button to visualize the model's behavior after training
if st.button('Render Environment'):
    # This assumes you have a method to capture and show the environment rendering.
    # For example, saving render frames and displaying them as a video or gif.
    st.write('This feature requires implementation.')
