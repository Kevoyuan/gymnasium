import gymnasium as gym
import numpy as np

# Initialize the environment
env = gym.make('Acrobot-v1')

# Q-learning parameters
learning_rate = 0.1
discount_rate = 0.95
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Simplify discretization for angles only - This is a significant simplification
angle_bins = 10  # Number of bins to discretize each angle into
velocity_bins = 10  # Simplified - might not use directly for discretization
q_table_shape = [angle_bins] * 4 + [velocity_bins] * 2 + [env.action_space.n]  # Adjust based on your discretization strategy
q_table = np.random.uniform(low=-1, high=0, size=q_table_shape)

def discretize_state(observation):
    """Discretizes the continuous state space."""
    # Extract angles and velocities
    cos1, sin1, cos2, sin2, vel1, vel2 = observation
    # Discretize angles
    angle1_bin = int((cos1 + 1) * angle_bins / 2)
    angle2_bin = int((sin1 + 1) * angle_bins / 2)
    angle3_bin = int((cos2 + 1) * angle_bins / 2)
    angle4_bin = int((sin2 + 1) * angle_bins / 2)
    # Simplification: Not discretizing velocities for this example
    # Construct the discrete state
    discrete_state = (angle1_bin, angle2_bin, angle3_bin, angle4_bin, 0, 0)  # Simplified
    return discrete_state

def step_environment(state, action):
    """Performs an action in the environment and applies custom reward logic."""
    next_state, reward, done, info = env.step(action)
    # Custom reward logic (if necessary)
    # Example: if not done: reward = -1; else: reward = 0
    next_discrete_state = discretize_state(next_state)
    return next_discrete_state, reward, done, info

def update_q_table(previous_state, state, action, reward, done):
    """Updates the Q-table using the learning and discount rates."""
    max_future_q = np.max(q_table[state])
    current_q = q_table[previous_state + (action,)]
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)
    q_table[previous_state + (action,)] = new_q if not done else reward  # Assign reward directly if done

def choose_action(state):
    """Chooses an action using an epsilon-greedy strategy."""
    if np.random.random() > exploration_rate:
        action = np.argmax(q_table[state])
    else:
        action = np.random.randint(0, env.action_space.n)
    return action
