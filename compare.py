import os
import numpy as np
import matplotlib.pyplot as plt
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv

# 比较传统调度算法与DDQN的调度性能
class TraditionalSchedulers:
    @staticmethod
    def round_robin(env):
        rewards = []
        for pod in range(env.num_pods):
            action = pod % env.num_nodes
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        return rewards

    @staticmethod
    def random_choice(env):
        rewards = []
        for pod in range(env.num_pods):
            action = np.random.choice(env.num_nodes)
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        return rewards

if __name__ == "__main__":
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # 加载DDQN模型
    model_path = "ddqn_model_95.weights.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file '{model_path}' not found.")

    agent = DDQNAgent(state_shape, action_size)
    agent.load(model_path)  # 确保这里使用的是保存的模型权重文件

    # 比较不同算法的表现
    ddqn_rewards = []
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])
        state = next_state
        ddqn_rewards.append(reward)

    rr_env = KubernetesEnv(num_nodes=5, num_pods=10)
    rr_rewards = TraditionalSchedulers.round_robin(rr_env)

    random_env = KubernetesEnv(num_nodes=5, num_pods=10)
    random_rewards = TraditionalSchedulers.random_choice(random_env)

    # 绘制每一步奖励值的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ddqn_rewards)), ddqn_rewards, label='DDQN', color='blue', marker='o')
    plt.plot(range(len(rr_rewards)), rr_rewards, label='Round Robin', color='orange', marker='o')
    plt.plot(range(len(random_rewards)), random_rewards, label='Random Choice', color='green', marker='o')

    # 在每个点上显示数值
    for i, value in enumerate(ddqn_rewards):
        plt.annotate(f'{value:.2f}', (i, ddqn_rewards[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, value in enumerate(rr_rewards):
        plt.annotate(f'{value:.2f}', (i, rr_rewards[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, value in enumerate(random_rewards):
        plt.annotate(f'{value:.2f}', (i, random_rewards[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Comparison of Scheduling Algorithms (Step by Step)')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_step_by_step.png")

    # 绘制总奖励值的柱状图
    total_ddqn_reward = np.sum(ddqn_rewards)
    total_rr_reward = np.sum(rr_rewards)
    total_random_reward = np.sum(random_rewards)

    rewards = [total_ddqn_reward, total_rr_reward, total_random_reward]
    labels = ['DDQN', 'Round Robin', 'Random Choice']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, rewards, color=['blue', 'orange', 'green'])

    # 在柱状图上显示数值
    for i, value in enumerate(rewards):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.xlabel('Schedulers')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Scheduling Algorithms (Total Reward)')
    plt.savefig("comparison_total_reward.png")
