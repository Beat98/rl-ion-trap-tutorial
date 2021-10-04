import numpy as np
import matplotlib.pyplot as plt

from PS.agent.Universal_Agent import UniversalAgent
from PS.ecm.Universal_ECM import UniversalECM
from ENV.IonTrap_env import IonTrapEnv

def run(adj_matrix, KWARGS, eta, beta, num_episodes):

    # initialize Environment
    env = IonTrapEnv(**KWARGS)

    #
    ecm = UniversalECM(gamma_damping=0, eta_glow_damping=eta, beta=beta)
    ag = UniversalAgent(ECM=ecm, actions=env.actions, adj_matrix=adj_matrix)

    data_steps = np.array([])

    for n in range(num_episodes):
        # initial observation from environment
        observation = env.reset()

        # bool: whether or not the environment has finished the episode
        done = False
        # int: the current time step in this episode
        num_steps = 0

        while not done:
            # increment counter
            num_steps += 1
            # predict action
            action = ag.step(observation)
            # perform action on environment and receive observation and reward
            observation, reward, done = env.step(action)

            ag.learn(reward)

            if done:
                data_steps = np.append(data_steps, num_steps)

    return data_steps

def initialize_adj_matrix(KWARGS):
    env = IonTrapEnv(**KWARGS)
    num_actions = len(env.actions)

    adj_matrix = np.zeros((num_actions + 1, num_actions + 1))
    adj_matrix[0][list(range(1, num_actions + 1))] = 1

    return adj_matrix

def rewarded_srv(num_ions, dim):
    srv = [dim for n in range(num_ions)]
    return srv

num_episodes = 500

dim = 3
num_ions = 2
max_steps = 10
phases = {'pulse_angles': [np.pi / 2], 'pulse_phases': [np.pi / 2], 'ms_phases': [-np.pi / 2]}
KWARGS = {'num_ions': num_ions, 'dim': dim, 'goal': [rewarded_srv(num_ions, dim)], 'phases': phases,
          'max_steps': max_steps}

eta = 0.25
beta = 0.3

adj_matrix = initialize_adj_matrix(KWARGS)
data_steps = run(adj_matrix, KWARGS, eta, beta, num_episodes)

plt.plot(data_steps)
plt.show()