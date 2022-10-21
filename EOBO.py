#
# Authors: K. Kosek-Szott
# AGH University of Science and Technology, Poland
#

from UORA import UORA
from Times import Times
from itertools import chain
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def get_results_static(episode):

    network_configurations = [[4, 4], [6, 4], [8, 4], [10, 4], [12, 4], [16, 8],
                              [18, 8], [20, 8], [22, 8],
                              [24, 8], [33, 16], [30, 16],
                              [28, 16], [27, 16], [28, 16],
                              [4, 4], [7, 4], [10, 4],
                              [13, 4], [16, 4], [15, 8],
                              [18, 8], [21, 8], [24, 8],
                              [27, 8], [32, 16], [21, 16],
                              [26, 16], [27, 16], [27, 16],
                              [4, 4], [9, 4], [14, 4], [19, 4],
                              [24, 4], [13, 8], [18, 8],
                              [23, 8], [28, 8], [33, 8],
                              [44, 16], [16, 16], [16, 16],
                              [16, 16], [16, 16]]

    for e in range(episode):
        thr_step = []
        col_step = []
        alpha_step = []
        efficiency_step = []
        n_sta = int(4)
        n_ru = int(4)
        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
            n_ru, n_sta, txop=txop)

        for i in range(len(network_configurations)):

            time1 = 0  # allows to set up a measurement interval
            time = 1  # allows to set up a measurement interval
            random.seed(1)

            collisions = 0
            tput = 0
            thr_s = {}  # per-station throughput
            pc_s = {}  # per-station collision probability
            del_s = {}  # per-station delay

            n_ru = network_configurations[i][1]
            n_sta = network_configurations[i][0]

            alpha1 = alpha
            counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
            alpha = alpha1

            j = 0
            while j < 10:
                time1 = 0
                j += 1
                collisions = 0
                tput = 0
                thr_s = {}
                pc_s = {}
                del_s = {}

                print("n_ru:{} n_sta: {}".format(n_ru, n_sta))

                while time1 < time:
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                    time1 += 0.01  # allows to modify E-OBO adjustment time

                for sta in range(0, n_sta):  # collect statistics
                    tput += thr_results['thr'][sta]
                    collisions += thr_results['colision_prob'][sta]
                    collisions_prob = collisions / n_sta
                    del_s[sta] = counters['delay_a'][sta]
                    thr_s[sta] = thr_results['thr'][sta]
                    pc_s[sta] = thr_results['colision_prob'][sta]

                # calculate general delay statistics
                filtered_vals = [v for _, v in del_s.items() if v != 0]
                if len(filtered_vals) == 0:
                    average_del = 0
                else:
                    average_del = sum(filtered_vals) / \
                        len(filtered_vals)  # average delay
                    max_del = np.max(filtered_vals)  # max delay
                    min_del = np.min(filtered_vals)  # min delay
                fairness_del = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))  # delay fairness
                del_f = fairness_del

                # calculate general throughput statistics
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

                # calculate general collision probability statistics
                filtered_vals = [v for _, v in pc_s.items() if v != 0]
                if len(filtered_vals) > 0:
                    average_pc = sum(filtered_vals) / len(filtered_vals)
                else:
                    average_pc = 0
                fairness_pc = np.power(
                    sum(filtered_vals), 2) / (n_sta*sum(np.power(filtered_vals, 2)))
                pc_f = fairness_pc

                # gather per-step results
                thr_step.append(tput)
                col_step.append(average_pc)
                alpha_step.append(alpha)
                efficiency_step.append(
                    tput/(n_ru*uora.times.transmission_rate/1e6))

                # write results to a csv file
                f = open("EOBO-results-static.csv", 'a',
                         encoding='UTF8', newline='',)
                writer = csv.writer(f, delimiter=',')
                writer.writerow([n_sta, tput, n_ru, average_pc, tput/(n_ru *
                                                                      Times().transmission_rate/1e6), thr_s, pc_s, average_del])
                f.close()

    results = [n_sta, tput, n_ru, collisions_prob, tput /
               (n_ru*uora.times.transmission_rate/1e6), thr_f, pc_f, average_del]
    return results, efficiency_step,  col_step, alpha_step


def get_results_dynamic(episode, base):

    max_steps = 20*base  # set the number of steps

    for e in range(episode):
        thr_step = []
        col_step = []
        alpha_step = []
        efficiency_step = []
        n_sta = int(4)
        n_ru = int(4)
        no_saturated = n_sta  # no of saturated stations
        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
            n_ru, n_sta, txop=txop)

        print("n_ru:{} n_sta: {}".format(n_ru, n_sta))

        for i in range(max_steps):
            time1 = 0
            time = 1
            random.seed(1)

            collisions = 0
            tput = 0
            thr_s = {}
            pc_s = {}
            del_s = {}

            if i > 0:
                if (i % (max_steps/3) == 0) & (i < (max_steps/1.5+1)):
                    n_ru *= 2
                    n_ru = int(min(n_ru, 32))
                    n_sta = int(max(n_ru, 2*n_ru-5))

                    print("n_ru:{} n_sta: {}".format(n_ru, n_sta))
                    alpha1 = alpha
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
                    alpha = alpha1
                    while time1 < time:
                        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                            n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                        time1 += 0.01

                if (i % (max_steps/3) == 0) & (i > (max_steps/1.5)):
                    n_ru /= 2
                    n_ru = int(max(n_ru, 4))
                    n_sta = int(max(n_ru, 2*n_ru-5))
                    print("n_ru:{} n_sta: {}".format(n_ru, n_sta))
                    alpha1 = alpha
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
                    alpha = alpha1
                    while time1 < time:
                        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                            n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                        time1 += 0.01

                if (i % (max_steps/15) == 0) & (i < (max_steps/1.5+1)):
                    n_sta += int(5)
                    print("n_ru:{} n_sta: {}".format(n_ru, n_sta))
                    alpha1 = alpha
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
                    alpha = alpha1
                    while time1 < time:
                        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                            n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                        time1 += 0.01

                elif (i % (max_steps/15) == 0) & (i > (max_steps/1.5)):
                    n_sta = int(max(n_ru, n_sta-5))
                    alpha1 = alpha
                    print("n_ru:{} n_sta: {}".format(n_ru, n_sta))
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
                    alpha = alpha1

                    while time1 < time:
                        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                            n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                        time1 += 0.01

                else:
                    n_sta = int(n_sta)
                    while time1 < time:
                        counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                            n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                        time1 += 0.01

            else:
                alpha1 = alpha
                counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                    n_ru, n_sta, first='false', counters={}, times_dict={}, thr_results={}, txop=False)
                alpha = alpha1
                while time1 < time:
                    counters, times_dict, thr_results, transmitted, alpha = uora.backoff_procedure(
                        n_ru, n_sta, 'true', counters, times_dict, thr_results,  txop=False)
                    time1 += 0.01

            collisions = 0
            tput = 0
            thr_s = {}
            pc_s = {}
            del_s = {}

            for sta in range(0, n_sta):
                tput += thr_results['thr'][sta]
                collisions += thr_results['colision_prob'][sta]
                collisions_prob = collisions / n_sta
                del_s[sta] = counters['delay_a'][sta]
                thr_s[sta] = thr_results['thr'][sta]
                pc_s[sta] = thr_results['colision_prob'][sta]

            filtered_vals = [v for _, v in del_s.items() if v != 0]
            if len(filtered_vals) == 0:
                average_del = 0
            else:
                average_del = sum(filtered_vals) / len(filtered_vals)
                max_del = np.max(filtered_vals)
                min_del = np.min(filtered_vals)
            fairness_del = np.power(sum(filtered_vals), 2) / \
                (n_sta*sum(np.power(filtered_vals, 2)))
            del_f = fairness_del

            filtered_vals = [v for _, v in thr_s.items() if v != 0]
            if len(filtered_vals) == 0:
                average_thr = 0
            else:
                average_thr = sum(filtered_vals) / len(filtered_vals)
                max_t = np.max(filtered_vals)
                min_t = np.min(filtered_vals)
            fairness_thr = np.power(sum(filtered_vals), 2) / \
                (n_sta*sum(np.power(filtered_vals, 2)))
            thr_f = fairness_thr

            filtered_vals = [v for _, v in pc_s.items() if v != 0]
            if len(filtered_vals) > 0:
                average_pc = sum(filtered_vals) / len(filtered_vals)
            else:
                average_pc = 0
            fairness_pc = np.power(sum(filtered_vals), 2) / \
                (n_sta*sum(np.power(filtered_vals, 2)))
            pc_f = fairness_pc

            thr_step.append(tput)
            col_step.append(average_pc)
            alpha_step.append(alpha)
            efficiency_step.append(
                tput/(n_ru*uora.times.transmission_rate/1e6))

            f = open("EOBO-results-dynamic.csv", 'a',
                     encoding='UTF8', newline='',)
            writer = csv.writer(f, delimiter=',')
            writer.writerow([n_sta, tput, n_ru, average_pc, tput/(n_ru *
                                                                  Times().transmission_rate/1e6), thr_f, pc_f, average_del])

            f.close()

    results = [n_sta, tput, n_ru, collisions_prob, tput /
               (n_ru*uora.times.transmission_rate/1e6), thr_s, pc_s, average_del]
    return results, efficiency_step,  col_step, alpha_step


if __name__ == '__main__':

    dynamic = 1  # defines which scenario is run, a static setting enables the static definition of scenarios

    if dynamic == 1:
        f = open("EOBO-results-dynamic.csv", 'a',
                 encoding='UTF8', newline='', )
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['no of stas', 'througput', 'no of RUs', 'p_col', 'efficiency',
                         'throughput-faifness', 'pc-fairness', 'avg_del', 'delay_min', 'delay_max'])
        f.close()

        f = open("EOBO-parameters-dynamic.txt",
                 'a', encoding='UTF8', newline='', )
        writer = csv.writer(f, delimiter=',')
        f.close()
    else:
        f = open("EOBO-results-static.csv", 'a', encoding='UTF8', newline='', )
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['no of stas', 'througput', 'no of RUs', 'p_col', 'efficiency',
                         'throughput-faifness', 'pc-fairness', 'avg_del', 'delay_min', 'delay_max'])
        f.close()

        f = open("EOBO-parameters-static.txt", 'a',
                 encoding='UTF8', newline='', )
        writer = csv.writer(f, delimiter=',')
        f.close()

    ep = 1
    txop = False
    n_sta = 4
    n_ru = 4

    if dynamic == 1:
        # the base should be set for dynamic scenario generation to increase the maximum number of stations in a network
        bases = [5, 15, 30]
    else:
        bases = [1]

    for base in bases:
        uora = UORA(n_sta, n_ru)
        if dynamic == 1:
            results, efficiency_step, col_step, alpha_step = get_results_dynamic(
                ep, base)  # get dynamic results
        else:
            results, efficiency_step, col_step, alpha_step = get_results_static(
                ep)  # get static results
        plt.plot([i for i in range(len(efficiency_step))], efficiency_step)
        plt.xlabel('steps')
        plt.ylabel('efficiency')
        plt.show()
        plt.plot([i for i in range(len(efficiency_step))], col_step)
        plt.xlabel('steps')
        plt.ylabel('pc')
        plt.show()
        plt.plot([i for i in range(len(efficiency_step))], alpha_step)
        plt.xlabel('steps')
        plt.ylabel('alpha')
        plt.show()

        if dynamic == 1:
            f = open("EOBO-parameters-dynamic.txt",
                     'a', encoding='UTF8', newline='',)
        else:
            f = open("EOBO-parameters-static.txt",
                     'a', encoding='UTF8', newline='',)
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['efficiency_step'])
        writer.writerow([efficiency_step])
        writer.writerow(['col_step'])
        writer.writerow([col_step])
        writer.writerow(['alpha_step'])
        writer.writerow([alpha_step])
        f.close()
