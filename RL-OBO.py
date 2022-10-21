#
# Authors: K. Kosek-Szott
# AGH University of Science and Technology, Poland
#
# ML part based on:
# https://colab.research.google.com/github/ehennis/ReinforcementLearning/blob/master/06-DDQN.ipynb#scrollTo=Fwq9n3K_ZOJ5
# https://keon.github.io/deep-q-learning/
# https://github.com/shivaverma/Orbit/blob/master/README.md

from UORA_RL import UORA
from Times import Times
from itertools import chain
import csv
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam

np.random.seed(0)


class DQN:

    """ Deep q learning """

    def __init__(self, action_space, state_space):
        # model parameters
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1  # exploration rate
        self.gamma = .95  # discount rate
        self.batch_size = 100
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):  # DQN model
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        targets = rewards + self.gamma * \
            (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode, steps):  # DQN training
    np.random.seed(0)

    loss = []
    thr_ep = []
    col_ep = []
    alpha_ep = []
    efficiency_ep = []

    action_space = 3  # size of the action space
    state_space = 1  # size of the state
    max_steps = steps  # number of steps

    agent = DQN(action_space, state_space)

    for e in range(episode):
        print("Episode:", e)

        n_sta = int(4)
        n_ru = int(4)
        print("n_ru:", n_ru)
        print("n_sta:", n_sta)

        # reset environment
        state, counters, times_dict, thr_results, alpha = env.reset(
            n_sta, n_ru)
        state = np.reshape(state, (1, state_space))
        score = 0

        # ML
        for i in range(max_steps):
            action = agent.act(state)
            time1 = 0
            time = 1

            if i > 0:
                # define the range of steps for which the number of STAs/RUs will be changed in the followin way
                if (i % 500 == 0) & (i < 2001):
                    n_ru *= 2  # increase the number of available RUs
                    n_ru = int(min(n_ru, 32))  # set the limit of max RUs
                    # set the number of stations in the network
                    n_sta = int(max(n_ru, 2*n_ru-5))

                    print("n_ru:", n_ru)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                if (i % 500 == 0) & (i > 2000):
                    n_ru /= 2
                    n_ru = int(max(n_ru, 4))
                    n_sta = int(max(n_ru, 2*n_ru-5))

                    print("n_ru:", n_ru)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                if (i % 100 == 0) & (i < 2001):
                    n_sta += int(5)

                    print("n_sta:", n_sta)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                elif (i % 100 == 0) & (i > 2000):
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    n_sta = int(max(n_ru, n_sta-5))

                    print("n_sta:", n_sta)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)
                else:
                    n_sta = int(n_sta)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)
            else:
                print("n_ru:", n_ru)
                reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                    action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

            score += round(reward)  # calculate score
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()  # replay

            # gather per-step results
            thr_ep.append(sum(thr_results['thr']))
            col_ep.append(sum(thr_results['colision_prob'])/n_sta)
            alpha_ep.append(alpha)
            efficiency_ep.append(
                sum(thr_results['thr'])/(n_ru*Times().transmission_rate/1e6))

            # analyze per-step results
            if i > 1:
                collisions = 0
                tput = 0
                thr_s = {}
                pc_s = {}
                del_s = {}

                for sta in range(0, n_sta):
                    tput += thr_results['thr'][sta]
                    collisions += thr_results['colision_prob'][sta]

                    if counters['send'][sta] == 0:
                        del_s[sta] = 0
                    else:
                        del_s[sta] = counters['delay_a'][sta] / \
                            counters['send'][sta]
                        thr_s[sta] = thr_results['thr'][sta]
                        pc_s[sta] = thr_results['colision_prob'][sta]

                # delay calculation
                filtered_vals = [v for _, v in del_s.items() if v != 0]
                if len(filtered_vals) == 0:
                    average_del = 0
                else:
                    average_del = sum(filtered_vals) / len(filtered_vals)
                fairness_del = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                del_f = fairness_del

                # throughput calculation
                filtered_vals = [v for _, v in thr_s.items() if v != 0]
                if len(filtered_vals) == 0:
                    average_thr = 0
                else:
                    average_thr = sum(filtered_vals) / len(filtered_vals)
                    max_t = np.max(filtered_vals)
                    min_t = np.min(filtered_vals)
                fairness_thr = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                thr_f = fairness_thr

                # collision probability calculation
                filtered_vals = [v for _, v in pc_s.items() if v != 0]
                if len(filtered_vals) > 0:
                    average_pc = sum(filtered_vals) / len(filtered_vals)
                else:
                    average_pc = 0
                fairness_pc = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                pc_f = fairness_pc

                # write results to a CSV file
                f = open("RL-OBO-results-training.csv",
                         'a', encoding='UTF8', newline='',)
                writer = csv.writer(f, delimiter=',')
                writer.writerow([n_sta, tput, n_ru, average_pc, tput/(n_ru *
                                                                      Times().transmission_rate/1e6), thr_f, pc_f, average_del])

                f.close()

        loss.append(score)

    return loss, thr_ep, col_ep, alpha_ep, efficiency_ep, agent


def test_dqn(episode, agent, base, steps):  # DQN testing
    np.random.seed(0)

    loss_test = []
    thr_ep_test = []
    col_ep_test = []
    alpha_ep_test = []
    efficiency_ep_test = []
    n_sta = int(4)
    n_ru = int(4)

    action_space = 3
    state_space = 1
    max_steps_test = steps

    for e in range(episode):
        n_sta = 4
        n_ru = 4
        print("n_ru:", n_ru)
        print("n_sta:", n_sta)

        state, counters, times_dict, thr_results, alpha = env.reset(
            n_sta, n_ru)
        state = np.reshape(state, (1, state_space))
        score = 0

        for i in range(max_steps_test):
            action = agent.act(state)
            time1 = 0
            time = 1
            random.seed(1)

            if i > 0:
                if (i % (max_steps_test/3) == 0) & (i < (max_steps_test/1.5+1)):
                    n_ru *= 2
                    n_ru = int(min(n_ru, 32))
                    n_sta = int(max(n_ru, 2*n_ru-random.randint(1, base)))

                    print("n_ru:", n_ru)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                if (i % (max_steps_test/3) == 0) & (i > (max_steps_test/1.5)):
                    n_ru /= 2
                    n_ru = int(max(n_ru, 4))
                    n_sta = int(max(n_ru, 2*n_ru-random.randint(1, base)))

                    print("n_ru:", n_ru)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                if (i % (max_steps_test/15) == 0) & (i < (max_steps_test/1.5+1)):
                    n_sta += int(random.randint(1, base))

                    print("n_sta:", n_sta)
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

                elif (i % (max_steps_test/15) == 0) & (i > (max_steps_test/1.5)):
                    counters, times_dict, thr_results, alpha = env.stepreset(
                        n_sta, n_ru, alpha)
                    n_sta = int(max(n_ru, 2*n_ru-random.randint(1, base)))

                    print("n_sta:", n_sta)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)
                else:
                    n_sta = int(n_sta)
                    reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                        action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)
            else:
                print("n_ru:", n_ru)
                counters, times_dict, thr_results, alpha = env.stepreset(
                    n_sta, n_ru, alpha)
                reward, next_state, done, counters, times_dict, thr_results, alpha = env.step(
                    action, n_ru, n_sta, counters, times_dict, thr_results, round(alpha, 1), 1)

            next_state = np.reshape(next_state, (1, state_space))
            state = next_state
            thr_ep_test.append(sum(thr_results['thr']))
            col_ep_test.append(sum(thr_results['colision_prob'])/n_sta)
            alpha_ep_test.append(alpha)
            efficiency_ep_test.append(
                sum(thr_results['thr'])/(n_ru*Times().transmission_rate/1e6))

            if i > 1:
                collisions = 0
                tput = 0
                thr_s = {}
                pc_s = {}
                del_s = {}

                for sta in range(0, n_sta):
                    tput += thr_results['thr'][sta]
                    collisions += thr_results['colision_prob'][sta]

                    if counters['send'][sta] == 0:
                        del_s[sta] = 0
                    else:
                        del_s[sta] = counters['delay_a'][sta] / \
                            counters['send'][sta]
                        thr_s[sta] = thr_results['thr'][sta]
                        pc_s[sta] = thr_results['colision_prob'][sta]

                # delay calculation
                filtered_vals = [v for _, v in del_s.items() if v != 0]
                if len(filtered_vals) == 0:
                    average_del = 0
                else:
                    average_del = sum(filtered_vals) / len(filtered_vals)
                fairness_del = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                del_s = fairness_del

                # throughput calculation
                filtered_vals = [v for _, v in thr_s.items() if v != 0]
                if len(filtered_vals) == 0:
                    average_thr = 0
                else:
                    average_thr = sum(filtered_vals) / len(filtered_vals)
                    max_t = np.max(filtered_vals)
                    min_t = np.min(filtered_vals)
                fairness_thr = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                thr_s = fairness_thr

                # collision probability calculation
                filtered_vals = [v for _, v in pc_s.items() if v != 0]
                if len(filtered_vals) > 0:
                    average_pc = sum(filtered_vals) / len(filtered_vals)
                else:
                    average_pc = 0
                fairness_pc = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                pc_s = fairness_pc

                # writing results to a csv file
                f = open("RL-OBO-results-testing.csv",
                         'a', encoding='UTF8', newline='',)
                writer = csv.writer(f, delimiter=',')
                writer.writerow([n_sta, tput, n_ru, average_pc, tput/(n_ru *
                                                                      Times().transmission_rate/1e6), thr_s, pc_s, average_del])
                f.close()

        loss_test.append(score)

    return loss_test, thr_ep_test, col_ep_test, alpha_ep_test, efficiency_ep_test


if __name__ == '__main__':

    f = open("RL-OBO-results-training.csv", 'a', encoding='UTF8', newline='', )
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['no of stas', 'througput', 'no of RUs', 'p_col', 'efficiency',
                     'throughput-faifness', 'pc-fairness', 'avg_del', 'delay_min', 'delay_max'])
    f.close()

    f = open("RL-OBO-parameters-training.txt",
             'a', encoding='UTF8', newline='', )
    writer = csv.writer(f, delimiter=',')
    f.close()

    f = open("RL-OBO-results-testing.csv", 'a', encoding='UTF8', newline='', )
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['no of stas', 'througput', 'no of RUs', 'p_col', 'efficiency',
                     'throughput-faifness', 'pc-fairness', 'avg_del', 'delay_min', 'delay_max'])
    f.close()

    f = open("RL-OBO-parameters-testing.txt",
             'a', encoding='UTF8', newline='', )
    writer = csv.writer(f, delimiter=',')
    f.close()

    txop = False  # turn off TXOP limit, TXOP limit support is left for future implementation
    n_sta = 0
    n_ru = 0
    ep = 20  # no of episodes for DQN training
    max_steps = 2000  # no of steps per episode
    reward = 0
    score = 0

    for n_sta in [4]:
        N_RU_array = [4]
        for n_ru in N_RU_array:
            env = UORA(n_ru, n_sta)
            loss, thr_ep, col_ep, alpha_ep, efficiency_ep, agent = train_dqn(
                ep, max_steps)

            # plotting
            plt.plot([i for i in range(len(efficiency_ep)-max_steps, len(efficiency_ep))],
                     efficiency_ep[(len(efficiency_ep)-max_steps):len(efficiency_ep)])
            plt.xlabel('steps')
            plt.ylabel('last ep efficiency')
            plt.show()

            plt.plot([i for i in range(len(efficiency_ep)-max_steps, len(efficiency_ep))],
                     col_ep[(len(efficiency_ep)-max_steps):len(efficiency_ep)])
            plt.xlabel('steps')
            plt.ylabel('last ep pcol')
            plt.show()

            plt.plot([i for i in range(len(efficiency_ep)-max_steps, len(efficiency_ep))],
                     alpha_ep[(len(efficiency_ep)-max_steps):len(efficiency_ep)])
            plt.xlabel('steps')
            plt.ylabel('last ep alpha')
            plt.show()

            plt.plot([i for i in range(len(loss))], loss)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.show()

            plt.plot([i for i in range(len(efficiency_ep))], efficiency_ep)
            plt.xlabel('steps')
            plt.ylabel('efficiency')
            plt.show()

            plt.plot([i for i in range(len(efficiency_ep))], col_ep)
            plt.xlabel('steps')
            plt.ylabel('pc')
            plt.show()

            plt.plot([i for i in range(len(efficiency_ep))], alpha_ep)
            plt.xlabel('steps')
            plt.ylabel('alpha')
            plt.show()

            # writing the results
            f = open("RL-OBO-parameters-training.txt",
                     'a', encoding='UTF8', newline='',)
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['max_steps'])
            writer.writerow([max_steps])
            writer.writerow(['efficiency_ep'])
            writer.writerow([efficiency_ep])
            writer.writerow(['col_ep'])
            writer.writerow([col_ep])
            writer.writerow(['alpha_ep'])
            writer.writerow([alpha_ep])
            writer.writerow(['loss'])
            writer.writerow([loss])
            f.close()

# %%
    for base in [5]:  # base for dynamic selection of the number of stations present in the network
        ep_test = 1  # no of episodes for DQN testing
        max_steps_test = 1500  # maximum number of steps
        loss_test, thr_ep_test, col_ep_test, alpha_ep_test, efficiency_ep_test = test_dqn(
            ep_test, agent, base, max_steps_test)

        # plotting
        plt.plot([i for i in range(len(efficiency_ep_test)-max_steps_test, len(efficiency_ep_test))],
                 efficiency_ep_test[(len(efficiency_ep_test)-max_steps_test):len(efficiency_ep_test)])
        plt.xlabel('steps')
        plt.ylabel('last ep efficiency')
        plt.show()

        plt.plot([i for i in range(len(efficiency_ep_test)-max_steps_test, len(efficiency_ep_test))],
                 col_ep_test[(len(efficiency_ep_test)-max_steps_test):len(efficiency_ep_test)])
        plt.xlabel('steps')
        plt.ylabel('last ep pcol')
        plt.show()

        plt.plot([i for i in range(len(efficiency_ep_test)-max_steps_test, len(efficiency_ep_test))],
                 alpha_ep_test[(len(efficiency_ep_test)-max_steps_test):len(efficiency_ep_test)])
        plt.xlabel('steps')
        plt.ylabel('last ep alpha')
        plt.show()

        plt.plot([i for i in range(len(loss_test))], loss_test)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.show()

        plt.plot([i for i in range(len(efficiency_ep_test))],
                 efficiency_ep_test)
        plt.xlabel('steps')
        plt.ylabel('efficiency')
        plt.show()

        plt.plot([i for i in range(len(efficiency_ep_test))], col_ep_test)
        plt.xlabel('steps')
        plt.ylabel('pc')
        plt.show()

        plt.plot([i for i in range(len(efficiency_ep_test))], alpha_ep_test)
        plt.xlabel('steps')
        plt.ylabel('alpha')
        plt.show()

        # writting the results
        f = open("RL-OBO-parameters-testing.txt",
                 'a', encoding='UTF8', newline='',)
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['max_steps_test'])
        writer.writerow([max_steps_test])
        writer.writerow(['efficiency_ep_test'])
        writer.writerow([efficiency_ep_test])
        writer.writerow(['col_ep_test'])
        writer.writerow([col_ep_test])
        writer.writerow(['alpha_ep_test'])
        writer.writerow([alpha_ep])
        writer.writerow(['loss_test'])
        writer.writerow([loss_test])
        f.close()
# %%
