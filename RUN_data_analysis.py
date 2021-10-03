import time
import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



height = 1080
aspect = 4 / 3
dpi = 200

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.top': 'on',
    'xtick.major.bottom': 'on',
    'ytick.major.left': 'on',
    'ytick.major.right': 'on',
    'xtick.top': True,
    'ytick.right': True})
mpl.rcParams['figure.figsize'] = (height * aspect / dpi, height / dpi)
plt.rcParams['figure.dpi'] = dpi

from agent.Universal_Agent import UniversalAgent
from ecm.Universal_ECM import UniversalECM
#from ecm.HValuesToProbabilities import linear_probabilities, softmax
from env.IonTrap_env import IonTrapEnv


def run(adj_matrix, KWARGS, eta, beta, num_episodes, avg):
    full_data_steps = np.zeros((avg, num_episodes))
    # full_reward_list = np.zeros((avg, num_episodes))

    laser_seq_list = []
    time_list = []

    for i in range(avg):

        start_time = time.time()

        # initialize Environment
        env = IonTrapEnv(**KWARGS)
        # linear_probabilities is a function that converts h-values to probabilities by simply normalizing them
        # gamma=0 means no forgetting, eta=1.0 means no interaction between steps
        ecm = UniversalECM(gamma_damping=0, eta_glow_damping=eta, beta=beta)
        ag = UniversalAgent(ECM=ecm, actions=env.actions, adj_matrix=adj_matrix)

        data_steps = np.array([])
        reward_list = np.array([])

        for n in range(num_episodes):
            # initial observation from environment
            observation = env.reset()

            # bool: whether or not the environment has finished the episode
            done = False
            # int: the current time step in this episode
            num_steps = 0
            cum_reward = 0
            action_seq = []
            laser_seq = []
            srv_seq = []
            while not done:
                # increment counter
                num_steps += 1
                # predict action

                action = ag.step(observation)

                #random:
                #action = np.random.choice(len(env.actions))


                if n == num_episodes - 1:
                    laser_seq = np.append(laser_seq, action)
                # perform action on environment and receive observation and reward
                observation, reward, done = env.step(action)

                srv_seq.append(env.srv(observation))

                cum_reward += reward

                ag.learn(reward)

                if done:
                    data_steps = np.append(data_steps, num_steps)
                    #print(srv_seq)
                    # reward_list = np.append(reward_list, cum_reward)

            if n == num_episodes - 1:
                laser_seq_list.append(laser_seq)
                end_time = time.time()
                time_list.append(end_time-start_time)

        full_data_steps[i, :] = data_steps
        # full_reward_list[i, :] = reward_list

    avg_data_steps = np.mean(full_data_steps, axis=0)
    std_data_steps = np.std(full_data_steps, axis=0)

    return avg_data_steps, std_data_steps, np.asarray(laser_seq_list), time_list


def initialize_config(config, num_ions, KWARGS):
    env = IonTrapEnv(**KWARGS)
    num_actions = len(env.actions)

    if config == 1:
        adj_matrix = np.zeros((num_actions + 1, num_actions + 1))
        adj_matrix[0][list(range(1, num_actions + 1))] = 1

    elif config == 2:
        adj_matrix = np.zeros((num_actions + 2, num_actions + 2))
        adj_matrix[-1][list(range(2, num_actions + 1))] = 1
        adj_matrix[0][[1, -1]] = 1

    elif config == 3:
        adj_matrix = np.zeros((num_actions + 1, num_actions + 1))
        adj_matrix[0][1] = 1

    else:
        print("invalid configuration")

    return adj_matrix


def rewarded_srv(num_ions, dim):
    srv = [dim for n in range(num_ions)]
    return srv


def store_data(data, config, eta, beta, path, data_name):
    np.savetxt(
        f"{path}{data_name}_config_{config}_dim_{dim}_ions_{num_ions}_eta_{eta}_beta_{beta}_episodes_{num_episodes}_avg_{avg}.txt",
        data)


def store_seq(seq, config, eta, beta, path, data_name):
    with open(f"{path}{data_name}_config_{config}_dim_{dim}_ions_{num_ions}_eta_{eta}_beta_{beta}_episodes_{num_episodes}_avg_{avg}.csv",'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=",")
        mywriter.writerows(seq)


def get_data_for_comparison(config_s, eta_s, beta_s, path):

    for eta in eta_s:
        for beta in beta_s:
            for config in config_s:

                #if eta == beta:
                    adj_matrix = initialize_config(config, num_ions, KWARGS)
                    avg_data_steps, std_data_steps, laser_seq_list, time_list = run(adj_matrix, KWARGS, eta, beta, num_episodes, avg)

                    # store data
                    store_seq(laser_seq_list, config, eta, beta, path, "Iontrap_final_laser_seq")
                    store_data(avg_data_steps, config, eta, beta, path, "Iontrap_avg_steps")
                    store_data(std_data_steps, config, eta, beta, path, "Iontrap_avg_steps_std")
                    store_data(time_list, config, eta, beta, path, "Iontrap_time_per_agent")

    return avg_data_steps, std_data_steps, laser_seq_list, time_list

def when_learned(data, limit):
    for i in range(len(data)):

        if i > limit:
            m = 0
            for n in range(i - limit, i):
                if data[n] == data[i]:
                    m += 1

            if m == limit:
                return i - limit
            else:
                continue



def plot_comparison(start, end, config_s, eta_s, beta_s, path, colors):
    xs = np.arange(start, end)

    table = np.zeros((32,4))
    i = 0
    n = 0

    for eta in eta_s:
        for beta in beta_s:
            for config in config_s:

                #if (eta != 0.15 and beta != 0.2):
                    data = np.loadtxt(
                        f"{path}Iontrap_avg_steps_config_{config}_dim_{dim}_ions_{num_ions}_eta_{eta}_beta_{beta}_episodes_{num_episodes}_avg_{avg}.txt")
                    err = np.loadtxt(
                        f"{path}Iontrap_avg_steps_std_config_{config}_dim_{dim}_ions_{num_ions}_eta_{eta}_beta_{beta}_episodes_{num_episodes}_avg_{avg}.txt")

                    learned = when_learned(data,50)

                    table[i,:] = [eta,beta,data[x_cut],learned]
                    i += 1


                    if data[x_cut] < y_cut:
                        color = colors[n]
                        n += 1
                        print(f"result: {data[x_cut]}")
                        print(f"error: {err[x_cut]}")
                        plt.plot(xs, data[start:end], color=color, linestyle='-',
                                 label=rf"$\eta$ = {eta}, $\beta$ = {beta}")#, config: {config}", ms=2)  #
                        # plt.plot(xs, data[start:end] + err[start:end], color=color, linestyle='--', alpha=0.2, ms=0.01,
                        #          lw=0.1)
                        # plt.plot(xs, data[start:end] - err[start:end], color=color, linestyle='--', alpha=0.2, ms=0.01,
                        #          lw=0.1)
                        # plt.fill_between(xs, data[start:end] + err[start:end], data[start:end] - err[start:end],
                        #                  color=color, alpha=0.1)
                        #plt.ylim([0, 50])

                        plt.legend()
                        plt.ylabel('average number of pulses')
                        plt.xlabel('episode')

    df = pd.DataFrame(table, columns = ["eta","beta","result","learned"])
    print(df.to_latex())
    plt.tight_layout(pad=0.1)
    #plt.savefig(f"figures/ion_trap_paramOpt_best_config_2")
    plt.show()


num_episodes = 500
avg = 1

dim = 3
num_ions = 2
max_steps = 10
phases = {'pulse_angles': [np.pi / 2], 'pulse_phases': [np.pi / 2], 'ms_phases': [-np.pi / 2]}
KWARGS = {'num_ions': num_ions, 'dim': dim, 'goal': [rewarded_srv(num_ions, dim)], 'phases': phases,
          'max_steps': max_steps}


eta_s = [0.1,0.15,0.2,0.25]
beta_s = [0.15,0.2,0.25,0.3]

eta_s = [0.25]
beta_s = [0.3]

config_s = [1]

path = "data/ion_trap_tree/"

x_cut = num_episodes-1
y_cut = 1000

colors = ["r","g","b","darkorange","y","k","grey", "olive","b","gold","lime","navy","brown","c","purple","hotpink","crimson",
          "r","g","darkorange","k","y","grey", "olive","b","gold","lime","navy","brown","c","purple","hotpink","crimson"]

avg_data_steps, std_data_steps, laser_seq_list, time_list = get_data_for_comparison(config_s, eta_s, beta_s, path)
plot_comparison(0, x_cut, config_s, eta_s, beta_s, path, colors)



