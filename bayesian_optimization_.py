import numpy as np
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv
from bayes_opt import BayesianOptimization

# 定义目标函数
def objective(alpha, beta, gamma, delta):
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DDQNAgent(state_shape, action_size)
    agent.load("ddqn_model_95.weights.h5")  # 确保这里使用的是保存的模型权重文件

    total_reward = 0
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_shape])
        state = next_state
        total_reward += reward

    return total_reward

# 定义优化参数的边界
pbounds = {
    'alpha': (0.01, 100.0),
    'beta': (0.01, 100.0),
    'gamma': (0.01, 100.0),
    'delta': (0.01, 100.0)
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

# 开始优化
optimizer.maximize(
    init_points=5,
    n_iter=25,
)

# 保存最优参数到文件
with open("best_params.txt", "a") as f:
    f.write(str(optimizer.max) + "__\n")
