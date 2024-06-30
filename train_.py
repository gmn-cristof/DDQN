import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv

# Function to plot training progress and save as images
def plot_training_progress_and_save(episodes, rewards, epsilons, save_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, label='Total Reward', marker='o')
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, epsilons, label='Epsilon', marker='o', color='orange')
    plt.title('Epsilon Decay')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot as an image
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DDQNAgent(state_shape, action_size)

    episodes = 1000
    batch_size = 32

    # Initialize lists to store metrics
    episode_rewards = []
    episode_epsilons = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, *state_shape])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, *state_shape])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.update_target_model()
        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)

        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")

        if agent.epsilon <= 0.01:
            agent.save(f"ddqn_model_{e}.weights.h5")  # Save the model
            print("Training stopped. Epsilon threshold reached.")
            break

        if e % 50 == 0:
            agent.save(f"ddqn_model_{e}.weights.h5")  # Save the model

    # Specify the path where you want to save the plot image
    save_path = "training_progress_plot.png"

    # Plotting the training progress and saving as an image
    plot_training_progress_and_save(range(len(episode_rewards)), episode_rewards, episode_epsilons, save_path)

