import gym
import numpy as np
import matplotlib.pyplot as plt

from PS.agent.Universal_Agent import UniversalAgent
from PS.ecm.Universal_ECM import UniversalECM

from gym_interface import OpenAIEnvironment

adj_matrix = np.zeros((2 + 1, 2 + 1))
adj_matrix[0][list(range(1, 2 + 1))] = 1

eta = 1
beta = 0.01

env = OpenAIEnvironment('CartPole-v1')
ecm = UniversalECM(gamma_damping=0, eta_glow_damping=eta, beta=beta)
ag = UniversalAgent(ECM=ecm, actions=env.num_actions_list, adj_matrix=adj_matrix)

data_steps = np.array([])
for i_episode in range(1000):
    observation = env.reset()
    n = 0
    for t in range(100):
        n += 1
        print(observation)
        action = ag.step(observation)
        observation, reward, done, info = env.step(action)
        ag.learn(reward)
        if done:
            data_steps = np.append(data_steps, n)
            break

plt.plot(data_steps)
plt.show()
