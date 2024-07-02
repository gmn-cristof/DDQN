import gym
from gym import spaces
import numpy as np

class KubernetesEnv(gym.Env):
    def __init__(self, num_nodes=5, num_pods=10):
        super(KubernetesEnv, self).__init__()

        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes, 7), dtype=np.float32)

        self.num_nodes = num_nodes
        self.num_pods = num_pods
        self.nodes = np.random.rand(num_nodes, 7)
        self.pods = self.generate_pods(num_pods)

        self.current_pod = 0

    def generate_pods(self, num_pods):
        pods = []
        for _ in range(num_pods):
            pod_type = np.random.choice(['cpu', 'gpu', 'io', 'network', 'average'])
            random_factor = np.random.rand()  # 生成一个随机因子
            if pod_type == 'cpu':
                pods.append(random_factor * np.random.rand(7) * [0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            elif pod_type == 'gpu':
                pods.append(random_factor * np.random.rand(7) * [0.5, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5])
            elif pod_type == 'io':
                pods.append(random_factor * np.random.rand(7) * [0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5])
            elif pod_type == 'network':
                pods.append(random_factor * np.random.rand(7) * [0.5, 0.5, 0.5, 0.5, 0.85, 0.5, 0.5])
            elif pod_type == 'average':
                pods.append(random_factor * np.random.rand(7) * [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        return np.array(pods)
    # 七维向量的各个维度：
    # CPU: 第一个元素，表示 CPU 资源需求。通常以 CPU 使用率或请求的 CPU 核数表示。
    # GPU: 第二个元素，表示 GPU 资源需求。通常以 GPU 使用率或请求的 GPU 核数表示。
    # 内存: 第三个元素，表示内存资源需求。通常以内存使用率或请求的内存大小表示。
    # IO: 第四个元素，表示输入/输出 (I/O) 资源需求。通常以 I/O 操作次数或 I/O 带宽表示。
    # 网络带宽: 第五个元素，表示网络带宽资源需求。通常以网络吞吐量或网络延迟表示。
    # 磁盘: 第六个元素，表示磁盘资源需求。通常以磁盘读写速率或磁盘容量表示。
    # 其他资源: 第七个元素，表示其他资源需求。这可以是任何特定于应用程序或环境的资源需求，如特殊硬件需求等。


    def reset(self):
        self.nodes = np.random.rand(self.num_nodes, 7)
        self.pods = self.generate_pods(self.num_pods)
        self.current_pod = 0
        return self.nodes

    def step(self, action):
        if action < 0 or action >= self.num_nodes:
            raise ValueError(f"Action {action} is out of bounds for {self.num_nodes} nodes")
        
        pod = self.pods[self.current_pod]
        self.nodes[action] += pod

        reward = self.compute_reward(action, pod)

        self.current_pod += 1
        done = self.current_pod >= self.num_pods

        return self.nodes, reward, done, {}

    def compute_reward(self, action, pod):
        # 超参数
        alpha = 10.0
        beta = 10.0
        gamma = 100.0
        delta = 64.0

        # 过载惩罚，当资源利用率超过1.0时
        overload_penalty = np.sum(self.nodes[self.nodes > 1.0] - 1.0)
        overload_reward = - alpha * overload_penalty  # 减少过载惩罚的权重

        # pod与节点的匹配度奖励
        match_reward = np.dot(pod, self.nodes[action])
        match_reward_value = beta * match_reward  # 增加正奖励的权重

        # 增加负载均衡的奖励，使所有节点的负载更加均衡
        balance_reward = - gamma * np.std(self.nodes, axis=0).sum()

        # 增加集群资源利用率的奖励
        total_utilization = np.mean(self.nodes, axis=0).sum()
        utilization_reward = delta * total_utilization

        reward = overload_reward + match_reward_value + balance_reward + utilization_reward

        # 打印表格
        # print(f"{'Parameter':<35} {'Value':<15}")
        # print("=" * 50)
        # print(f"{'Alpha (Overload Penalty Weight)':<35} {alpha:<15}")
        # print(f"{'Beta (Match Reward Weight)':<35} {beta:<15}")
        # print(f"{'Gamma (Balance Reward Weight)':<35} {gamma:<15}")
        # print(f"{'Delta (Utilization Reward Weight)':<35} {delta:<15}")
        # print("-" * 50)
        # print(f"{'Overload Penalty':<35} {overload_penalty:<15.4f}")
        # print(f"{'Overload Reward':<35} {overload_reward:<15.4f} (Alpha * Overload Penalty)")
        # print("-" * 50)
        # print(f"{'Match Reward':<35} {match_reward:<15.4f}")
        # print(f"{'Match Reward Value':<35} {match_reward_value:<15.4f} (Beta * Match Reward)")
        # print("-" * 50)
        # print(f"{'Balance Reward':<35} {balance_reward:<15.4f} (Gamma * Std Dev of Nodes)")
        # print("-" * 50)
        # print(f"{'Total Utilization':<35} {total_utilization:<15.4f}")
        # print(f"{'Utilization Reward':<35} {utilization_reward:<15.4f} (Delta * Total Utilization)")
        # print("=" * 50)
        # print(f"{'Total Reward':<35} {reward:<15.4f}")
        # print("\n")

        return reward



    def render(self, mode='human'):
        pass
