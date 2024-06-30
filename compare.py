import numpy as np
import matplotlib.pyplot as plt
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv

# 比较传统调度算法与DDQN的调度性能
class TraditionalSchedulers:
    @staticmethod
    def round_robin(env):
        total_reward = 0
        for pod in range(env.num_pods):
            action = pod % env.num_nodes
            _, reward, _, _ = env.step(action)
            total_reward += reward
        return total_reward

    @staticmethod
    def random_choice(env):
        total_reward = 0
        for pod in range(env.num_pods):
            action = np.random.choice(env.num_nodes)
            _, reward, _, _ = env.step(action)
            total_reward += reward
        return total_reward

if __name__ == "__main__":
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    # 加载DDQN模型
    agent = DDQNAgent(state_shape, action_size)
    agent.load("ddqn_model_95.weights.h5.weights.h5")  # 确保这里使用的是保存的模型权重文件

    # 比较不同算法的表现
    ddqn_reward = 0
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])
        state = next_state
        ddqn_reward += reward

    rr_env = KubernetesEnv(num_nodes=5, num_pods=10)
    rr_reward = TraditionalSchedulers.round_robin(rr_env)

    random_env = KubernetesEnv(num_nodes=5, num_pods=10)
    random_reward = TraditionalSchedulers.random_choice(random_env)

    # 绘制比较图
    rewards = [ddqn_reward, rr_reward, random_reward]
    labels = ['DDQN', 'Round Robin', 'Random Choice']

    plt.bar(labels, rewards, color=['blue', 'orange', 'green'])
    plt.xlabel('Schedulers')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Scheduling Algorithms')
    plt.savefig("comparison.png")
