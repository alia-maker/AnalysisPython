import numpy as np
import datetime
import pandas as pd
import enum
# from scipy.stats import linregress
from HoltWinters import HoltWinters

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class Status(enum.Enum):
    obs = 0
    miss = 1


class Mark(enum.Enum):
    discrete_interpolated = 1
    serial_interpolated = 2
    serial_extrapolated = 3
    discrete_extrapolated = 4


class TimeSeriesAnalysis(object):
    def __init__(self, data, number_value_per_hour, period_day):
        self.data = data
        self.number_value_per_hour = number_value_per_hour
        self.period_day = period_day


    def balancing_data(self):
        self.m = []
        '''m: измерительная мера'''

        DateTime_new = []

        interval_value = np.round(np.linspace(0, 60, self.number_value_per_hour), 0).astype(int)
        max_dif = int((interval_value[1] - interval_value[0])) * 60

        dif_interval = np.abs(interval_value - self.data.index[0].minute)
        argmin_value = np.argmin(dif_interval)

        first_time = datetime.datetime(year=self.data.index[0].year, month=self.data.index[0].month, day=self.data.index[0].day)

        if interval_value[argmin_value] < 60:
            a = datetime.timedelta(hours=self.data.index[0].hour, minutes=int(interval_value[argmin_value]))
        else:
            a = datetime.timedelta(hours=self.data.index[0].hour + 1, minutes=0)

        first_time = first_time.__add__(a)
        DateTime_new.append(first_time)

        temp_time = first_time
        while temp_time < first_time.__add__(datetime.timedelta(days=self.period_day)):
            temp_time = temp_time.__add__(datetime.timedelta(minutes=int(60 / self.number_value_per_hour)))
            DateTime_new.append(temp_time)

        # добавляем первое значение, и для него равные ближайшие значения во избежание ошибки
        i = 0
        Value_new = []
        Value_new.append(float(self.data.values[i]))
        self.m.append(0)
        self.RealNearestValues = []
        self.RealNearestValues.append([i, i])

        i = 0
        for timestamp in DateTime_new[1:]:
            similar_values = []
            k = i
            # while True:
            #     similar_values.append(self.data.index[k])
            #     if self.data.index[k] > timestamp:
            #         break
            #     k += 1

            while self.data.index[k] < timestamp:
                similar_values.append(self.data.index[k])
                # if self.data.index[k] > timestamp:
                #     break
                k += 1

            # print('timestamp:', timestamp, 'similar_values:', similar_values)


            dif_values = []
            for v in similar_values:
                dif_values.append(timestamp - v)

            argmin_v = int(np.argmin(dif_values))

            # indices = np.array(dif_values).argsort()
            # if dif_values.__len__() > 0:
            self.RealNearestValues.append([i + argmin_v, k])
            # else:
            #     self.RealNearestValues.append([i + k, i + k])

            # print(indices)
            # argmin_v = int(np.argmin(dif_values))
            # print('timestamp:', timestamp, 'min dif seconds:', dif_values[argmin_v].seconds)
            dif_prev_timestamp = dif_values[argmin_v]
            dif_next_timestamp = self.data.index[k] - timestamp
            if dif_prev_timestamp.days == 0 and dif_prev_timestamp.seconds < max_dif\
                    and dif_next_timestamp.days == 0 and dif_next_timestamp.seconds < max_dif:
                # if len(indices) > 1:
                # if self.data.index[i + indices[0]] < self.data.index[i + indices[1]]:
                y = self.__linear_interpolation(timestamp.minute, self.data.index[i + argmin_v].minute,
                                                self.data.index[k].minute, self.data.values[i + argmin_v],
                                                self.data.values[k])
                # else:
                #     y = self.__linear_interpolation(timestamp.minute, self.data.index[i + indices[1]].minute,
                #                                     self.data.index[i + indices[0]].minute,
                #                                     self.data.values[i + indices[1]],
                #                                     self.data.values[i + indices[0]])
                # else:
                #     y = float(self.data.values[i + indices[0]])
                Value_new.append(y)
                self.m.append(0)
            else:
                Value_new.append(0)
                self.m.append(2)

            i += argmin_v
            # if i < 0:
            #     i = 0


        # print('number of zeros', Value_new.count(0))
        # print(Value_new)
        # print(DateTime_new.__len__())


        # for r in range(len(Value_new)):
        #     print(Value_new[r],DateTime_new[r], self.data.index[self.RealNearestValues[r][0]],
        #                         self.data.index[self.RealNearestValues[r][1]],
        #                         self.data.values[self.RealNearestValues[r][0]],
        #                         self.data.values[self.RealNearestValues[r][1]],
        #         )
        # print(Value_new, DateTime_new)
        # exit()
        self.balanced_data = pd.DataFrame(data=Value_new, index=DateTime_new)

    def __analis_interval(self, i, i_miss, i_obs, prev_i_obs):
        if i_miss == 1 and i_obs > i_miss and prev_i_obs > i_miss:
            self.discrete_interpolated.append(i - 1 - i_obs)
            self.marker[i - i_obs - 1] = Mark.discrete_interpolated.value
        elif i_miss == 1 and (i_obs <= i_miss or prev_i_obs <= i_miss):
            self.discrete_extrapolated.append(i - 1 - i_obs)
            self.marker[i - 1 - i_obs] = Mark.discrete_extrapolated.value

        elif i_miss > 1 and i_obs > i_miss and prev_i_obs > i_miss:
            self.marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_interpolated.value
            self.serial_interpolated.append([i - i_miss - i_obs, i - i_obs])
        elif i_miss > 1 and (i_obs <= i_miss or prev_i_obs <= i_miss):
            self.serial_extrapolated.append([i - i_miss - i_obs, i - i_obs])
            self.marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_extrapolated.value

    def data_analysis(self):
        self.marker = np.zeros(self.m.__len__())
        self.discrete_interpolated = []
        self.serial_interpolated = []
        self.serial_extrapolated = []
        self.discrete_extrapolated = []
        state = Status.miss.value
        i_obs = 0
        prev_i_obs = 0
        i_miss = 0


        for i in range(len(self.m)):
            if Status.miss.value == state:
                if self.m[i] > 0:
                    i_miss += 1
                else:
                    state = Status.obs.value
                    ###
                    prev_i_obs = i_obs
                    i_obs = 0

            if Status.obs.value == state:
                if self.m[i] == 0:
                    i_obs += 1
                else:

                    state = Status.miss.value
                    ###
                    # self.__analis_interval(i, i_miss, i_obs)
                    ###
                    self.__analis_interval(i, i_miss, i_obs, prev_i_obs)
                    # if i_miss == 1 and i_obs > i_miss:
                    #     self.discrete_interpolated.append(i - 1 - i_obs)
                    #     self.marker[i - i_obs - 1] = Mark.discrete_interpolated.value
                    # elif i_miss == 1 and i_obs <= i_miss:
                    #     self.discrete_extrapolated.append(i - 1 - i_obs)
                    #     self.marker[i - 1 - i_obs] = Mark.discrete_extrapolated.value
                    #
                    # elif i_miss > 1 and i_obs > i_miss:
                    #     self.marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_interpolated.value
                    #     self.serial_interpolated.append([i - i_miss - i_obs, i - i_obs])
                    # elif i_miss > 1 and i_obs <= i_miss:
                    #     self.serial_extrapolated.append([i - i_miss - i_obs, i - i_obs])
                    #     self.marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_extrapolated.value
                    i_miss = 1
                    ###
                    # i_obs = 0

        i = len(self.m)
        if Status.obs.value == state:
            self.__analis_interval(i, i_miss, i_obs, prev_i_obs)




        # for t in range(self.m.__len__()):
        #     print(self.balanced_data.values[t], self.m[t], self.marker[t])
        # exit()
        # print('discrete_interpolated:', discrete_interpolated)
        # print('serial_interpolated:', serial_interpolated)
        # print('serial_extrapolated:', serial_extrapolated)
        # print('discrete_extrapolated', discrete_extrapolated)
        # return discrete_interpolated, discrete_extrapolated, serial_interpolated, serial_extrapolated

    def __linear_interpolation(self, x, x1, x2, y1, y2):
        if x1 > x2:
            x1 -= 60
        # print(x1, x2, y1, y2)
        # x = data.index[timestamps[0]].minute
        if x > x2:
            x -= 60

        a = (y2 - y1)/(x2 - x1)
        b = y1 - a * x1
        y = a * x + b
        # print(y)
        return float(y)


    def liquidation_discrete_values(self):
        # for index in range(self.balanced_data.__len__()):
        #     print(index, self.balanced_data.values[index])
        for i in self.discrete_interpolated:
            # print(i)
            # print('index:', index, 'changed value:', self.balanced_data.values[index])
            y = self.__linear_interpolation(self.balanced_data.index[i].minute,
                                            self.data.index[self.RealNearestValues[i][0]].minute,
                                            self.data.index[self.RealNearestValues[i][1]].minute,
                                            self.data.values[self.RealNearestValues[i][0]],
                                            self.data.values[self.RealNearestValues[i][1]]
                                            )
            self.balanced_data.values[i] = y
            self.m[i] = 0
        for i in self.discrete_extrapolated:
            y = self.__linear_interpolation(self.balanced_data.index[i].minute,
                                            self.data.index[self.RealNearestValues[i][0]].minute,
                                            self.data.index[self.RealNearestValues[i][1]].minute,
                                            self.data.values[self.RealNearestValues[i][0]],
                                            self.data.values[self.RealNearestValues[i][1]]
                                            )
            self.balanced_data.values[i] = y
            self.m[i] = 0


    def liquidation_series_interpolated_values(self):
        for pair in self.serial_interpolated:
            # print(pair)
            pull = []
            # print('with 4 values before and after: ', self.balanced_data.values[pair[0]-4:pair[1]+4])
            # print(self.marker[pair[0]:pair[1]])
            length = (pair[1] - pair[0])
            if list(self.m[pair[0] - length: pair[0]]).count(2) == 0 and list(self.m[pair[1]: pair[1] + length]).count(2) == 0:
                y = []
                y.extend(self.balanced_data.values[pair[0] - length: pair[0]])
                y.extend(self.balanced_data.values[pair[1]: pair[1] + length])

                y = np.array([float(elem) for elem in y])

                # print('left+right parts TS: y =', y)
                for i in range(0, self.balanced_data.__len__() - length * 3):
                    if list(self.m[i: i + 3 * length]).count(2) == 0:
                        x = []
                        x.extend(self.balanced_data.values[i: i + length])
                        x.extend(self.balanced_data.values[i + 2 * length: i + 3 * length])
                        x = np.array([float(elem) for elem in x])
                        pull.append(np.corrcoef(y, x)[0, 1])
                    else:
                        pull.append(0)
                # print('max cor coef:', np.max(pull))
                while True:
                    k = np.argmax(pull)
                    # print(k)
                    x = []
                    x.extend(self.balanced_data.values[k: k + length])
                    x.extend(self.balanced_data.values[k + 2 * length: k + 3 * length])
                    x = np.array([float(elem) for elem in x])
                    A = np.vstack([x, np.ones(len(y))]).T
                    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

                    break
                    # pull.pop(int(k))

                # print('the similarest TS:', x)
                # print('a =', a, 'b =', b)
                # print(self.balanced_data.values[k:k + 3 * length])
                t = (a * self.balanced_data.values[k + length:k + 2 * length] + b)
                # print('input values', t)
                # print()
                # exit()
                # for j in range(length):
                self.balanced_data.values[pair[0]:pair[1]] = t
                for p in range(pair[0], pair[1]):
                    self.m[p] = 0
            # print('data:',self.balanced_data.values[pair[0]-3:pair[1]+3])
            # print(self.balanced_data.values[k: k + 3 * length])
            # exit()3
    def liquidation_series_extrapolated_values(self):
        if self.serial_extrapolated.__len__() > 1:
            while self.serial_interpolated.__len__() == 0:
                last_missing_index = self.serial_extrapolated[0][1]
                distantion = []
                for pair in self.serial_extrapolated[1:]:
                    distantion.append(pair[0] - last_missing_index)
                    last_missing_index = pair[1]
                # print('serial_extrapolated misses', self.serial_extrapolated)
                # print('serial_interpolated', self.serial_interpolated)
                # print(distantion)
                argmin_dist = int(np.argmin(distantion))
                # self.serial_extrapolated[argmin_dist][1] = self.serial_extrapolated[argmin_dist + 1][1]
                # self.serial_extrapolated.pop(argmin_dist + 1)
                # print(self.serial_extrapolated)
                for i in range(self.serial_extrapolated[argmin_dist][1], self.serial_extrapolated[argmin_dist + 1][0]):
                    self.m[i] = 2
                self.data_analysis()
        else:
            self.balanced_data = self.balanced_data[:self.serial_extrapolated[0][0]]
            # self.serial_extrapolated.pop(0)
            self.m = self.m[:self.serial_extrapolated[0][0]]
        self.data_analysis()

    def find_period(self, x):
        # вектор ошибок
        errors = []
        # data = dataset2
        values = self.balanced_data.values
        # for v in values:
        #     print(v)
        # print(values.__len__())
        alpha, beta, gamma = x

        # задаём число фолдов для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=3)

        # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
        Errors = []
        for s in range(1, 150):
            for train, test in tscv.split(values):
                # print('train', train)
                # print('test', test)
                # exit()

                model = HoltWinters(series=values[train], slen=s, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
                model.triple_exponential_smoothing()

                predictions = model.result[-len(test):]
                actual = values[test]
                error = mean_squared_error(predictions, actual)
                errors.append(error)
            Errors.append(np.sum(errors))

        # Возвращаем средний квадрат ошибки по вектору ошибок
            print('s:', s, 'errors:', errors)
            errors = []
        Errors = np.array(Errors).T
        print(Errors.argsort())

        # print(np.argmin(Errors[0]), np.argmin(Errors[1]), np.argmin(Errors[2]))
        # exit()
        # return np.mean(np.array(errors))

    def timeseriesCVscore(self, x):
        # вектор ошибок
        errors = []
        # data = dataset2
        values = self.balanced_data.values
        # for v in values:
        #     print(v)
        # print(values.__len__())
        alpha, beta, gamma = x
        # print('x:', x)

        # задаём число фолдов для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=15)

        # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
        # Errors = []
        # for s in range(1, 150):
        for train, test in tscv.split(values):
            # print('len train:', len(train))
            # print('test', test)
            # exit()

            model = HoltWinters(series=values[train], slen=8, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
            model.triple_exponential_smoothing()

            predictions = model.result[-len(test):]
            actual = values[test]
            error = mean_squared_error(predictions, actual)
            errors.append(error)
            # Errors.append(np.sum(errors))

        # Возвращаем средний квадрат ошибки по вектору ошибок
        # print('errors:', errors)
            # errors = []
        # Errors = np.array(Errors).T
        # print(Errors.argsort())

        # print(np.argmin(Errors[0]), np.argmin(Errors[1]), np.argmin(Errors[2]))
        # exit()
        return np.mean(np.array(errors))

