import streamlit as st
from rl_environment.acrobot import AcrobotAgent
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image, ImageSequence


def create_gif_with_pillow(frames, filepath='animation.gif', total_duration_ms=4000):
    if not frames:
        raise ValueError("The frames list is empty.")

    num_frames = len(frames)
    # Calculate the duration each frame is shown to fit the total_duration_ms
    duration_per_frame_ms = total_duration_ms / num_frames  # Duration in milliseconds
    
    # Convert numpy array frames to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    # Save the frames as a GIF
    pil_images[0].save(
        filepath,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_per_frame_ms,  # Duration for each frame
        loop=0
    )

st.title('Acrobot Reinforcement Learning')

# Sidebar for configuration
st.sidebar.header('Configuration')
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)
discount_factor = st.sidebar.slider('Discount Factor', min_value=0.8, max_value=0.99, value=0.95, step=0.01)
episodes = st.sidebar.slider('Episodes', min_value=10, max_value=1000, value=100)
st.text(f'Learning Rate: {learning_rate}, Discount Factor: {discount_factor}, Episodes: {episodes}')
agent = AcrobotAgent(learning_rate=learning_rate, discount_factor=discount_factor)
if st.sidebar.button('Train Agent'):
    with st.spinner('Training...'):
        
        rewards = agent.train(episodes=episodes)  # This method now returns a list of rewards per episode

    # Plotting the rewards
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')
    st.pyplot(fig)


# Optional: Provide a button to visualize the model's behavior after training
# if st.button('Render Environment'):
#     with st.spinner('Rendering...'):
#         frames = agent.render(episodes=1)  # Capture frames from one episode
#         st.image(frames[-1], caption='Last frame of the episode', use_column_width=True)


if st.button('Animate!'):
    with st.spinner('Creating Animation for the First Episode...'):
        # Render and create a GIF for the first episode
        frames_first = agent.render(episodes=1)
        gif_path_first = "temp/acrobot_animation_first.gif"
        create_gif_with_pillow(frames_first, filepath=gif_path_first, total_duration_ms=1000)
        st.image(gif_path_first, caption='Acrobot Animation for the First Episode')
    
    # Check if the total number of episodes selected is more than 1
    if episodes > 1:
        with st.spinner(f'Creating Animation for the Last Episode...'):
            # Render the last episode. Since render() starts from the beginning every time,
            # and you want to skip to the last episode, you can use a dummy render call.
            # This is not efficient and should be optimized.
            # The agent.render() should be modified to directly jump to the desired episode,
            # but here's a quick workaround:
            for _ in range(episodes - 1):  # Dummy loop to advance the environment state
                agent.env.reset()
                while True:
                    _, _, done, _, _ = agent.env.step(agent.env.action_space.sample())
                    if done:
                        break
            
            # Now render the last episode after the dummy loop
            frames_last = agent.render(episodes=1)
            gif_path_last = f"temp/acrobot_animation_last.gif"
            create_gif_with_pillow(frames_last, filepath=gif_path_last, total_duration_ms=1000)
            st.image(gif_path_last, caption=f'Acrobot Animation for the Last Episode')