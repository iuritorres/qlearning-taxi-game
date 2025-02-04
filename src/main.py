"""Main module for the Taxi-v3 environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from utils.q_table import load_q_table, save_q_table


def run(episodes, is_training=True, render=False):
    """Run the Taxi-v3 environment."""
    env = gym.make(
        "Taxi-v3",
        render_mode="human" if render else None,
    )

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Q-table 500 x 6
    else:
        q = load_q_table("taxi.pkl")

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[new_state, :])
                    - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)

    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])

    plt.plot(sum_rewards)
    plt.savefig("taxi.png")

    if is_training:
        save_q_table(q, "taxi.pkl")


if __name__ == "__main__":
    run(5000)
    run(10, is_training=False, render=True)

