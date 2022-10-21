#
# Author: K. Domino
# AGH University of Science and Technology, Poland
#

import math


class Times:
    def __init__(self, mcs=2):
        self.MCS = {
            0: [7.3, 14.6, 30.6, 61.3],
            1: [14.6, 29.3, 61.3, 122.5],
            2: [21.9, 43.9, 91.9, 183.8],
            3: [29.3, 58.5, 122.5, 245],
            4: [43.9, 87.8, 183.8, 367.5],
            5: [58.5, 117, 245, 490],
            6: [65.8, 131.6, 275.6, 551.5],
            7: [73.1, 146.3, 306.3, 612.6],
            8: [87.8, 175.5, 367.5, 735],
            9: [97.5, 195, 408.3, 816.6],
            10: [109.7, 219.4, 459.4, 918.8],
            11: [121.9, 243.8, 510.4, 1020.8],
        }
        self.timeout = 16e-6  # s
        self.preamble = 40e-6  # s
        self.tf = 100e-6  # s
        self.mba = 68e-6  # s
        self.sifs = 16e-6  # s
        self.slot_time = 9e-6  # s
        self.txop = 3.844e-3  # s
        self.mac_header = 320  # bits
        self.hesu_preamble = 40e-6
        self.tail_bits = 18  # bits
        self.ofdm_symbol_duration = 16e-6  # s
        self.transmission_rate = 6.67e6  # bit/s
        self.frame_size = 10000  # bits
        self.service_field = 8  # bits

    def transmission_time(self, txop=False):
        """
        time spent when at least one STA transmit in at least one RU
        :param self:
        :param txop:
        :return:
        """
        ppdu = self.ppdu_duration()
        if not txop:
            time_spent = self.tf + 2 * self.sifs + ppdu + self.mba
        else:
            time_spent = self.tf + 3 * self.sifs + self.txop + self.mba
        return time_spent

    def non_transmission_time(self):
        """
        Time duration spent when no STAs transmit after TF
        :param self:
        :return:
        """
        time_wait = self.timeout + self.tf
        return time_wait

    def ppdu_duration(self):
        data_duration = self.hesu_preamble + \
            ((self.service_field + self.mac_header +
              self.frame_size + self.tail_bits) / self.transmission_rate)
        return data_duration
