#
# Authors: K. Domino (UORA), K. Kosek-Szott (ML)
# AGH University of Science and Technology, Poland
#
# ML part based on:
# https://colab.research.google.com/github/ehennis/ReinforcementLearning/blob/master/06-DDQN.ipynb#scrollTo=Fwq9n3K_ZOJ5
# https://keon.github.io/deep-q-learning/
# https://github.com/shivaverma/Orbit/blob/master/README.md

import random
import numpy as np
from Times import Times
np.seterr(divide='ignore', invalid='ignore')


class UORA():
    def __init__(self, n_sta, n_ru):
        self.ShortRetryLimit = 7
        self.LongRetryLimit = 4
        self.ocwmax = 31
        self.ocwmin = 7
        self.sim_time = 1000
        self.times = Times()
        self.reward = 0
        self.counters = {}
        self.times_dict = {}
        self.thr_results = {}

    def generate_obo_counter(self, retransmission_times=0, previous_ocw=0, ocwm=31, ocwmin=7):
        if retransmission_times == 0:
            ocw = ocwmin
            obo = random.randint(0, ocw)
        elif retransmission_times > 0:
            ocw = 2*(previous_ocw)+1
            if ocw >= ocwm:
                ocw = ocwm
            obo = random.randint(0, ocw)
        return obo, ocw

    def backoff_procedure(self, n_ru, n_sta, alpha=1, reward_success=3, reward_collision=2, reward_empty=1.5, first='false', counters={}, times_dict={}, thr_results={}, txop=False):
        transmission = 0
        collission = 0
        if first == 'false':
            counters = {}
            times_dict = {}
            thr_results = {}
            results_all = [0]
            tx_time = 0
            time_spent = 0
            time_wait = 0
            alpha = alpha
            empty_ru = []
            successfull_ru = []
            unsuccessfull_ru = []
            send_frames_counter = np.zeros(int(n_sta))
            collision_probability = np.zeros(int(n_sta))
            retransmission = np.zeros(int(n_sta))
            ocwm = np.ones(int(n_sta))*31
            ocwmin = np.ones(int(n_sta))*7
            saturation = np.zeros(int(n_sta))
            collision_cnt = np.zeros(int(n_sta))
            delay_cnt = np.zeros(int(n_sta))
            delay_all = np.zeros(int(n_sta))
            delay_avg = np.zeros(int(n_sta))
            new_counter = np.zeros((1, int(n_sta)))
            ocw_table = np.zeros((1, int(n_sta)))
            throughput = 0
            for i in range(0, n_sta):
                new_counter[0][i], ocw_table[0][i] = self.generate_obo_counter()
        else:
            send_frames_counter = counters['send'][:int(n_sta)]
            collision_probability = counters['pcol'][:int(n_sta)]
            retransmission = counters['retr'][:int(n_sta)]
            ocwmin = counters['ocwmin'][:int(n_sta)]
            ocwm = counters['ocwm'][:int(n_sta)]
            new_counter = counters['new_cnt']
            ocw_table = counters['ocw']
            saturation = counters['saturation'][:int(n_sta)]
            collision_cnt = counters['collision'][:int(n_sta)]
            delay_cnt = counters['delay'][:int(n_sta)]
            delay_all = counters['delay_a'][:int(n_sta)]
            delay_avg = counters['delay_avg'][:int(n_sta)]

            # serve new comming stations
            if n_sta > send_frames_counter.size:
                collision_probability = np.append(
                    collision_probability, np.zeros(n_sta-collision_probability.size))
                send_frames_counter = np.append(
                    send_frames_counter, np.zeros(n_sta-send_frames_counter.size))
                retransmission = np.append(
                    retransmission, np.zeros(n_sta-retransmission.size))
                ocwmin = np.append(ocwmin, np.zeros(n_sta-ocwmin.size))
                ocwm = np.append(ocwm, np.zeros(n_sta-ocwm.size))
                new_counter = np.append(new_counter, np.zeros(
                    n_sta-new_counter.size))*np.ones((1, n_sta))
                ocw_table = np.append(ocw_table, np.zeros(
                    n_sta-ocw_table.size))*np.ones((1, n_sta))
                saturation = np.append(
                    saturation, np.zeros(n_sta-saturation.size))
                collision_cnt = np.append(
                    collision_cnt, np.zeros(n_sta-collision_cnt.size))
                delay_cnt = np.append(
                    delay_cnt, np.zeros(n_sta-delay_cnt.size))
                delay_all = np.append(
                    delay_all, np.zeros(n_sta-delay_all.size))
                delay_avg = np.append(
                    delay_avg, np.zeros(n_sta-delay_avg.size))
                if (n_sta-new_counter.size) > 0:
                    for i in range(new_counter.size, n_sta-new_counter.size):
                        new_counter[0][i], ocw_table[0][i] = self.generate_obo_counter(
                        )
            else:
                new_counter1 = np.zeros((1, n_sta))
                new_counter1[0] = new_counter[0][:n_sta]
                new_counter = new_counter1
                ocw_table1 = np.zeros((1, n_sta))
                ocw_table1[0] = ocw_table[0][:n_sta]
                ocw_table = ocw_table1

            tx_time = times_dict['tx']
            time_spent = times_dict['spent']
            time_wait = times_dict['wait']
            empty_ru = counters['empty_ru']
            results_all = counters['results_all']
            successfull_ru = counters['successfull_ru']
            unsuccessfull_ru = counters['unsuccessfull_ru']
            alpha = alpha

        # calculate ppdu time
        ppdu_time = self.times.ppdu_duration()
        # check how many ppdus can be transmitted within next txop
        frame_no = max(1, int(self.times.txop / ppdu_time))

        # initiate RU-STA array
        ru_sta_array = np.zeros((n_ru, n_sta))

        # select new obo counter
        for sta in range(0, n_sta):
            new_counter[0][sta] = new_counter[0][sta] - alpha * n_ru

        # check if there are stations with OBO <=0
        null_obo_indices = np.where(new_counter <= 0)[1]

        # select ru for transmission
        for i in null_obo_indices:
            chosen_ru = random.randint(0, n_ru-1)
            ru_sta_array[chosen_ru][i] = 1
        for b in range(0, ru_sta_array.shape[0]):
            # check which RUs are occupied by transmissions
            d = np.where((ru_sta_array[:][b]) == 1)
            # check which RUs are empty
            e = np.where((ru_sta_array[:][b]) == 0)

            if len(e[0]) == ru_sta_array.shape[1]:
                empty_ru.append(1)
                self.reward -= reward_empty  # penalty for empty RUs
            else:
                empty_ru.append(0)

            # check if there are RUs in which only a single station transmits
            if len(d[0]) == 1:
                successfull_ru.append(1)
                self.reward += reward_success  # reward for successful RUs
            else:
                successfull_ru.append(0)

            # check if there are any unsuccessful RUs
            if len(d[0]) > 1:
                unsuccessfull_ru.append(1)
                self.reward -= reward_collision  # penalty for unsuccessful RUs
            else:
                unsuccessfull_ru.append(0)

            if len(d[0]) == 1:  # successful tranmission/successful RU
                transmission = 1  # signalize successful transmission
                retransmission[d] = 0
                if not txop:
                    for sta in d[0]:
                        send_frames_counter[sta] += 1
                        delay_all[sta] += delay_cnt[sta]
                        delay_avg[sta] = delay_all[sta] / \
                            send_frames_counter[sta]
                        delay_cnt[sta] = 0
                    for sta in e[0]:
                        delay_cnt[sta] += (self.times.transmission_time())
                else:
                    for sta in d[0]:
                        send_frames_counter[sta] += frame_no
                    for sta in e[0]:
                        delay_cnt[sta] += (self.times.transmission_time())
                new_counter[0][d], ocw_table[0][d] = self.generate_obo_counter(
                    0, 0, ocwm[sta], ocwmin[sta])

            elif len(d[0]) > 1:  # collision/unsuccessful RU
                for sta in d[0]:
                    retransmission[sta] += 1
                    delay_cnt[sta] += (self.times.transmission_time())
                    collision_cnt[sta] += 1
                    new_counter[0][sta], ocw_table[0][sta] = self.generate_obo_counter(
                        retransmission[sta], ocw_table[0][sta], ocwm[sta], ocwmin[sta])
                for sta in e[0]:
                    delay_cnt[sta] += (self.times.transmission_time())

            else:  # empty RU

                for sta in e[0]:
                    delay_cnt[sta] += (self.times.non_transmission_time())

        for b in range(0, ru_sta_array.shape[0]):
            if sum(ru_sta_array[:][b]) > n_ru:
                collission = 1  # signalize collision

        # calculate times
        if transmission == 1:
            tx_tranmission = self.times.transmission_time(txop)
            time_spent += tx_tranmission
            tx_time += tx_tranmission
            retransmission[d] = 0
        elif collission == 1:
            tx_tranmission = self.times.transmission_time(txop)
            time_spent += tx_tranmission
            tx_time += tx_tranmission
        else:
            tx_nontransmission = self.times.non_transmission_time()
            tx_time += tx_nontransmission
            time_wait += tx_nontransmission

        # remember the results
        counters['retr'] = retransmission
        counters['pcol'] = collision_probability
        counters['new_cnt'] = new_counter
        counters['send'] = send_frames_counter
        counters['ocw'] = ocw_table
        counters['collision'] = collision_cnt
        counters['saturation'] = saturation
        counters['delay'] = delay_cnt
        counters['delay_a'] = delay_all
        counters['delay_avg'] = delay_avg
        times_dict['tx'] = tx_time
        times_dict['spent'] = time_spent
        times_dict['wait'] = time_wait
        simulation_time = tx_time
        counters['ocwm'] = ocwm
        counters['ocwmin'] = ocwmin
        counters['empty_ru'] = empty_ru
        counters['results_all'] = results_all
        counters['successfull_ru'] = successfull_ru
        counters['unsuccessfull_ru'] = unsuccessfull_ru

        # calculate collision probability
        if ~np.isnan(any(collision_cnt)) == True:
            for sta in range(0, n_sta):
                if (collision_cnt[sta] + send_frames_counter[sta]) == 0:
                    collision_probability[sta] = 0
                else:
                    collision_probability[sta] = collision_cnt[sta] / \
                        (collision_cnt[sta] + send_frames_counter[sta])
        else:
            collision_probability = 0

        # calculate success probability
        if any(send_frames_counter) > 0:
            success_probability = send_frames_counter / \
                (collision_cnt + send_frames_counter)
        else:
            success_probability = 0

        # calculate throughput
        throughput = (send_frames_counter*self.times.frame_size)/(tx_time*1e6)

        # update throughput results
        thr_results['thr'] = throughput
        thr_results['colision_prob'] = collision_probability
        thr_results['success'] = success_probability

        if(tx_time >= self.sim_time):
            self.done = 1

        return counters, times_dict, thr_results, alpha

    # ----------ML part--------------
    # Possible actions:
    # 0 increase alpha
    # 1 do nothing
    # 2 decrease alpha

    def reset(self, n_sta, n_ru, reward_success, reward_collision, reward_empty):
        counters, times_dict, thr_results, alpha = self.backoff_procedure(
            n_ru, n_sta, 1, reward_success, reward_collision, reward_empty, 'false', counters={}, times_dict={}, thr_results={}, txop=False)
        state = [0]
        return state, counters, times_dict, thr_results, alpha

    def stepreset(self, n_sta, n_ru, alpha, reward_success, reward_collision, reward_empty):
        counters, times_dict, thr_results, alpha = self.backoff_procedure(
            n_ru, n_sta, alpha, reward_success, reward_collision, reward_empty, 'false', counters={}, times_dict={}, thr_results={}, txop=False)
        return counters, times_dict, thr_results, alpha

    def step(self, action, n_ru, n_sta, reward_success, reward_collision, reward_empty, counters, times_dict, thr_results, alpha, time):
        self.reward = 0
        self.done = 0
        time1 = 0  # allows to change the interval after which the impact of new alpha value on network performance is checked

        if action == 1:
            alpha = round(min(3, alpha+0.1), 2)
            self.reward -= 0.1

        if action == 2:
            alpha = round(max(0.1, alpha-0.1), 2)
            self.reward -= 0.1

        # check how the new alpha value impacts the network performance
        while time1 < time:
            counters, times_dict, thr_results, alpha = self.backoff_procedure(
                n_ru, n_sta,  alpha, reward_success, reward_collision, reward_empty, 'true', counters, times_dict, thr_results)
            time1 += 0.1

        # calculate state
        state = [round(sum(counters['unsuccessfull_ru'])/(sum(counters['unsuccessfull_ru']
                                                              )+sum(counters['successfull_ru'])+sum(counters['empty_ru'])), 2)]

        return self.reward, state, self.done, counters, times_dict, thr_results, alpha
