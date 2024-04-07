import streamlit as st
from rl_environment import acrobot  # Import your acrobot environment setup
from PIL import Image

def app():
    st.title('Acrobot Environment with Q-learning')

    # Ensure all necessary session state variables are initialized upfront
    if 'exploration_rate' not in st.session_state:
        st.session_state.exploration_rate = acrobot.exploration_rate
    if 'state' not in st.session_state or 'discrete_state' not in st.session_state:
        reset_environment()

    # Reset Environment button
    if st.button('Reset Environment'):
        reset_environment()

    # Take Step button
    if st.button('Take Step') and not st.session_state.done:
        take_step()

    # Render Environment button
    if st.button('Render Environment'):
        render_environment()

    # Display total reward
    st.text(f"Total Reward: {st.session_state['total_reward']}")

def reset_environment():
    st.session_state.state = acrobot.env.reset()
    st.text(f"Initial State: {st.session_state.state}")
    st.session_state.discrete_state = acrobot.discretize_state(st.session_state.state)
    st.text(f"Initial Discrete State: {st.session_state.discrete_state}")
    st.session_state.done = False
    st.session_state.total_reward = 0

def take_step():
    action = acrobot.choose_action(st.session_state.discrete_state)
    new_state, reward, done, _ = acrobot.step_environment(st.session_state.state, action)
    acrobot.update_q_table(st.session_state.discrete_state, new_state, action, reward, done)
    st.session_state.state = new_state
    st.session_state.discrete_state = acrobot.discretize_state(new_state)
    st.session_state.done = done
    st.session_state.total_reward += reward
    st.session_state.exploration_rate = max(acrobot.min_exploration_rate, acrobot.exploration_rate * acrobot.exploration_decay_rate)
    # Update the display
    update_display(action, reward, done)

def render_environment():
    img = Image.fromarray(acrobot.env.render(mode='rgb_array'))
    st.image(img, caption='Current State of the Acrobot Environment')

def update_display(action, reward, done):
    # Display updated information
    st.write(f"Action Taken: {action}, Reward: {reward}, Done: {done}")

if __name__ == '__main__':
    app()
