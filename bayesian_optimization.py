import numpy as np
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv
from bayes_opt import BayesianOptimization

# 定义超参数优化的目标函数
def ddqn_objective(alpha, beta, gamma, delta):
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DDQNAgent(state_shape, action_size)
    
    # 使用这些超参数设置环境
    env.alpha = alpha
    env.beta = beta
    env.gamma = gamma
    env.delta = delta

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

# 设置贝叶斯优化的参数范围
pbounds = {
    'alpha': (1.0, 20.0),
    'beta': (1.0, 20.0),
    'gamma': (10.0, 200.0),
    'delta': (32.0, 128.0)
}

optimizer = BayesianOptimization(
    f=ddqn_objective,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=10,
    n_iter=50,
)

print("最优超参数: ", optimizer.max)

# 保存最优的超参数
with open("best_params.txt", "a") as f:
    f.write(str(optimizer.max)+"\n")
