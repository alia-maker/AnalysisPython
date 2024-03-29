import numpy as np

class HoltWinters:

    """
    Модель Хольта-Винтерса с методом Брутлага для детектирования аномалий
    https://fedcsis.org/proceedings/2012/pliks/118.pdf

    # series - исходный временной ряд
    # slen - длина сезона
    # alpha, beta, gamma - коэффициенты модели Хольта-Винтерса
    # n_preds - горизонт предсказаний
    # scaling_factor - задаёт ширину доверительного интервала по Брутлагу (обычно принимает значения от 2 до 3)

    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.test_l = int(self.n_preds / 3 * 2)
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            if self.series[i] != np.inf and self.series[i + self.slen] != np.inf:
                sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = np.zeros(self.slen)
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)

        # print("n_seasons", n_seasons)
        # print(self.series)
        # print(type(self.series))
        # вычисляем сезонные средние
        for j in range(n_seasons):
            s = 0
            for i in range (self.slen*j,self.slen*j+self.slen):
                # season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
                if self.series[i] != np.inf:
                    s += self.series[i]
            season_averages.append(s/self.slen)
        # вычисляем начальные значения
        # print('slen', self.slen)
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):
                if self.series[self.slen * j + i] != np.inf:
                    sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()
        # print('seasonals', seasonals)

        for i in range(len(self.series)+self.n_preds):
        # for i in range(len(self.series)+self.test_l):
            if i == 0: # инициализируем значения компонент
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                self.PredictedDeviation.append(0)

                self.UpperBond.extend((self.result[0] +
                                      self.scaling_factor *
                                      self.PredictedDeviation[0]).tolist())

                self.LowerBond.extend((self.result[0] -
                                      self.scaling_factor *
                                      self.PredictedDeviation[0]).tolist())
                continue
            if i >= len(self.series): # прогнозируем
                # print('Smooth:', smooth, self.Smooth[-1])
                # exit()

                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i % self.slen])
                # print('i>len series', (smooth + m*trend) + seasonals[i % self.slen])
                # exit()
                # во время прогноза с каждым шагом увеличиваем неопределенность
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01)

            else:
                if self.series[i] != np.inf:
                    # print('i<len series')
                    val = self.series[i]
                    last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                    # print('last smoth',last_smooth, 'smooth', smooth)
                    trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                    # print('trend', trend)
                    seasonals[i % self.slen] = self.gamma * (val-smooth) + (1-self.gamma) * seasonals[i % self.slen]
                    # print(seasonals)
                    # print(self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen])
                    self.result.extend(smooth+trend+seasonals[i%self.slen])

                    # print('result append', smooth+trend+seasonals[i % self.slen])
                    # Отклонение рассчитывается в соответствии с алгоритмом Брутлага
                    self.PredictedDeviation.append(self.gamma * np.abs(val - self.result[i])
                                                   + (1-self.gamma)*self.PredictedDeviation[-1])

                    # print(self.result[-1])
                    # print(self.PredictedDeviation[-1])
                    # print(self.result[-1] +
                    #                       self.scaling_factor *
                    #                       self.PredictedDeviation[-1])

                    self.UpperBond.append(self.result[-1] +
                                          self.scaling_factor *
                                          self.PredictedDeviation[-1])
                    # exit()
                    self.LowerBond.append(self.result[-1] -
                                          self.scaling_factor *
                                          self.PredictedDeviation[-1])

                    self.Smooth.append(smooth)
                    self.Trend.append(trend)
                    self.Season.append(seasonals[i % self.slen])
                else:
                    pass
                    # print("inf value!!!!!")
                    # self.result.append(np.inf)
                    # self.PredictedDeviation.append(np.inf)
                    # self.UpperBond.append(np.inf)
                    # self.LowerBond.append(np.inf)

        # for i in range(len(self.Smooth)):
        #     print(i, self.Smooth[i], self.Trend[i])
        # exit()
        # print(self.LowerBond)
        # print(self.UpperBond)
        # print(self.PredictedDeviation)



