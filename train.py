import numpy as np
import tensorflow as tf
from ddqn_agent import DDQNAgent
from environment import KubernetesEnv

# 使用GPU训练
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    env = KubernetesEnv(num_nodes=5, num_pods=10)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DDQNAgent(state_shape, action_size)

    episodes = 1000
    batch_size = 32

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
        print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")

        if agent.epsilon <= 0.01:
            agent.save(f"ddqn_model_{e}.weights.h5")  # 保存模型
            print("Training stopped. Epsilon threshold reached.")
            break

        if e % 50 == 0:
            agent.save(f"ddqn_model_{e}.weights.h5")  # 保存模型

