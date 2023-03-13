import math
import numpy as np


class Channel:
    def __init__(self, subBand=10e6):
        self.L0 = 61094
        self.P_w_r = self.pNoise(26, 10e6)  # 白噪声功率
        self.n0 = math.pow(10, -20)  # w/Hz
        self.subBand = subBand  # 单位： Hz
        self.alpha = 2.75

    def pNoise(self, T, deitaF):
        return 1.38*math.pow(10, -23)*(273.15+T)*deitaF

    def distance(self, s: np.array, d: np.array):
        return np.linalg(s - d)

    # 发送功率， 通信距离
    def sinr(self, p, d, b):
        # pwr = self.pNoise(26, b * self.subBand)  # 白噪声
        pwr = self.n0 * b * self.subBand
        return (p/(self.L0*(math.pow(d, self.alpha))))/(pwr)

    # 返回通信速率
    def rate(self, p, d, b):
        rate = b * self.subBand * \
            math.log(1 + self.sinr(p, d, b))/(8*math.pow(2, 20))  # MB
        return rate
