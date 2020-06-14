import numpy as np
import datetime
import pandas as pd
import enum

# from scipy.stats import linregress
from HoltWinters import HoltWinters
# import sklearn.model_selection._split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
# from web_plotly import plotly_df

class Status(enum.Enum):
    obs = 0
    miss = 1


class Mark(enum.Enum):
    discrete_interpolated = 1
    serial_interpolated = 2
    serial_extrapolated = 3
    discrete_extrapolated = 4


class TimeSeriesAnalysis(object):
    def __init__(self, data, number_value_per_hour, period_day, change_length):
        self.data = data
        self.number_value_per_hour = number_value_per_hour + 1
        self.period_day = period_day
        self.season_len = 1
        self.change_length = change_length

    def balancing_data(self):
        self.m = []
        '''m: измерительная мера'''
        Timestamps_new = []
        new_timestamps_per_hour = np.round(np.linspace(0, 60, self.number_value_per_hour), 0).astype(int)
        dif_minutes = int((new_timestamps_per_hour[1] - new_timestamps_per_hour[0]))
        dif_interval = np.abs(new_timestamps_per_hour - self.data.index[0].minute)
        argmin_value = np.argmin(dif_interval)

        first_timestamps = datetime.datetime(year=self.data.index[0].year, month=self.data.index[0].month,
                                       day=self.data.index[0].day)

        if new_timestamps_per_hour[argmin_value] < 60:
            a = datetime.timedelta(hours=self.data.index[0].hour, minutes=int(new_timestamps_per_hour[argmin_value]))
        else:
            a = datetime.timedelta(hours=self.data.index[0].hour + 1, minutes=0)

        first_timestamps = first_timestamps.__add__(a)
        Timestamps_new.append(first_timestamps)

        temp_timestamp = first_timestamps

        while temp_timestamp < self.data.index[-1]:
            temp_timestamp = temp_timestamp.__add__(datetime.timedelta(minutes=dif_minutes))
            Timestamps_new.append(temp_timestamp)
        # print("new timestamps were created, length TS:", len(Timestamps_new))
        # добавляем первое значение, и для него равные ближайшие значения во избежание ошибки
        i = 0
        new_values = []
        new_values.append(float(self.data.values[i]))
        self.m.append(0)
        self.RealNearestValues = []
        self.RealNearestValues.append([i, i])
        k = 1
        for timestamp in Timestamps_new[1:]:
            k = k - 1

            while self.data.index[k] < timestamp:
                if k < len(self.data) - 1:
                    k += 1
                else:
                    break



            self.RealNearestValues.append([k - 1, k])

            dif_prev_timestamp = timestamp - self.data.index[k - 1]
            dif_next_timestamp = self.data.index[k] - timestamp

            if dif_prev_timestamp.days == 0 and dif_prev_timestamp.seconds < dif_minutes * 60\
                    and dif_next_timestamp.days == 0 and dif_next_timestamp.seconds < dif_minutes * 60:
                y = self.__linear_interpolation(timestamp.minute, self.data.index[k - 1].minute,
                                                self.data.index[k].minute, self.data.values[k - 1],
                                                self.data.values[k])
                # else:
                #     y = self.__linear_interpolation(timestamp.minute, self.data.index[i + indices[1]].minute,
                #                                     self.data.index[i + indices[0]].minute,
                #                                     self.data.values[i + indices[1]],
                #                                     self.data.values[i + indices[0]])
                # else:
                #     y = float(self.data.values[i + indices[0]])
                new_values.append(y)
                self.m.append(0)
            else:
                new_values.append(np.inf)
                self.m.append(2)
        print("значения получены")

        while new_values[-1] == np.inf:
            # if self.change_length==True:
            len_new_values = new_values.__len__()
            new_values = new_values[:len_new_values - 1]
            Timestamps_new = Timestamps_new[:len_new_values - 1]
            self.m = self.m[:len_new_values - 1]
            # elif self.change_length==False:
            #     new_values[-1] = new_values[-2]
            #     print("last repeat")

        self.balanced_data = pd.DataFrame(data=new_values, index=Timestamps_new)

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
                    prev_i_obs = i_obs
                    i_obs = 0

            if Status.obs.value == state:
                if self.m[i] == 0:
                    i_obs += 1
                else:

                    state = Status.miss.value
                    self.__analis_interval(i, i_miss, i_obs, prev_i_obs)
                    i_miss = 1

        i = len(self.m)
        if Status.obs.value == state:
            self.__analis_interval(i, i_miss, i_obs, prev_i_obs)
        # else:
        #     self.__analis_interval(i, i_miss, i_obs, prev_i_obs)


    def __linear_interpolation(self, x, x1, x2, y1, y2):
        if x1 > x2:
            x1 -= 60

        if x > x2:
            x -= 60
        if x1 != x2:
            a = (y2 - y1)/(x2 - x1)
        else:
            a = y2 - y1
        b = y1 - a * x1
        y = a * x + b
        return float(y)


    def liquidation_discrete_values(self):

        while self.m[-1] == 2:
            end = len(self.balanced_data.values)
            self.balanced_data = self.balanced_data[:int(end-1)]
            self.m = self.m[:int(end-1)]

        for i in self.discrete_interpolated:
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
            pull = []
            length = (pair[1] - pair[0])

            y = []
            y.extend(self.balanced_data.values[pair[0] - length: pair[0]])
            ###
            # print(pair[0], pair[1])
            for i in range(pair[0], pair[1]):
                if self.m[i]==1:
                    # print('m==1 i:', i)
                    y.append(self.balanced_data.values[i])
            y.extend(self.balanced_data.values[pair[1]: pair[1] + length])

            y = np.array([float(elem) for elem in y])
            # print('y inf aroud miss values:', list(y).count(np.inf))
            # print('len y:', len(y))
            available_size = False
            for i in range(0, self.balanced_data.__len__() - length * 3):
                if list(self.m[i: i + 3 * length]).count(2) == 0:
                    available_size = True
                    x = []
                    x.extend(self.balanced_data.values[i: i + length])
                    for j in range(pair[0], pair[1]):
                        if self.m[j] == 1:
                            x.append(self.balanced_data.values[i + length + j - pair[0]])
                            # print('m==1 x i:', i + length + j - pair[0])


                    x.extend(self.balanced_data.values[i + 2 * length: i + 3 * length])
                    x = np.array([float(elem) for elem in x])
                    # print('len x:', len(x))
                    # print('x inf number', list(x).count(np.inf))
                    corr = np.abs(np.corrcoef(y, x)[0, 1])
                    pull.append(corr)
                    if corr > 0.999:
                        break
                else:
                    pull.append(0)
            # print(pull)

            pull_without_nan = []
            for e in pull:
                if np.isnan(e):
                    pull_without_nan.append(0)
                else:
                    pull_without_nan.append(e)
                # print(e, type(e))
                # i
                #     pull_without_nan.append(0)
                # else:
                #     pull_without_nan.append(e)
            # print('без nan:',pull_without_nan)
            k = np.argmax(pull_without_nan)
            print('k', k, pull[k])
            x = []
            x.extend(self.balanced_data.values[k: k + length])
            for j in range(pair[0], pair[1]):
                if self.m[j] == 1:
                    x.append(self.balanced_data.values[k + length + j - pair[0]])
                    # print('m==1 x i:', k + length + j - pair[0])
            x.extend(self.balanced_data.values[k + 2 * length: k + 3 * length])
            x = np.array([float(elem) for elem in x])
            # print('best x number inf', list(x).count(np.inf))
                # while True:
            if pull[k]>0:
                A = np.vstack([x, np.ones(len(y))]).T
                a, b = np.linalg.lstsq(A, y, rcond=None)[0]
                    # if a != 0 or b != 0:
                    #     break
                    # else:
                    #     x = x[:int(x.__len__()*0.95)]
                    #     y = y[:int(y.__len__()*0.95)]

                t = a * self.balanced_data.values[k + length:k + 2 * length] + b
                self.balanced_data.values[pair[0]:pair[1]] = t
                for p in range(pair[0], pair[1]):
                    self.m[p] = 0
            else:
                f = True
                i = 0
                while f == True and i < len(pull):
                    if np.isnan(pull[i]):
                        k = i
                        # print('i=', i)
                        t = self.balanced_data.values[k + length:k + 2 * length]
                        self.balanced_data.values[pair[0]:pair[1]] = t
                        for p in range(pair[0], pair[1]):
                            self.m[p] = 0
                        f = False
                    i += 1

            # else:
            #     self.liquidation_series_extrapolated_values()



    def liquidation_series_extrapolated_values(self):
        # print()
        # print('balanced_data', self.balanced_data.__len__)
        #
        # print('m', self.m)
        # print('serial_extrapolated',self.serial_extrapolated)
        # избавляемся от пропусков, близких к краю
        # if self.serial_extrapolated.__len__()>0:
        #     while self.serial_extrapolated[0][1] - self.serial_extrapolated[0][0] > self.serial_extrapolated[0][0]:
        #         self.balanced_data = self.balanced_data[self.serial_extrapolated[0][1] + 1:]
        #         self.m = self.m[self.serial_extrapolated[0][1] + 1:]
        #         self.data_analysis()


        while True:
            if self.serial_extrapolated.__len__() == 0:
                break
            if self.serial_extrapolated[0][1] - self.serial_extrapolated[0][0] > self.serial_extrapolated[0][0]:
                if self.change_length==True:
                    self.balanced_data = self.balanced_data[self.serial_extrapolated[0][1] + 1:]
                    self.m = self.m[self.serial_extrapolated[0][1] + 1:]
                    # self.data_analysis()
                elif self.change_length == False:
                    # print("Use linear interpolated method for keeping length")
                    count = self.serial_extrapolated[0][1] - self.serial_extrapolated[0][0] + 1
                    len_between_points = self.balanced_data.index[self.serial_extrapolated[0][1] + 1].minute - \
                                         self.balanced_data.index[self.serial_extrapolated[0][0] - 1].minute
                    gamma = len_between_points/count
                    current_minute = self.data.index[self.serial_extrapolated[0][0] - 1].minute
                    for t in range(self.serial_extrapolated[0][0], self.serial_extrapolated[0][1] + 1):
                        # print('before', self.balanced_data.values[t])
                        current_minute += gamma
                        y = self.__linear_interpolation(current_minute,
                                                        self.data.index[self.serial_extrapolated[0][0] - 1].minute,
                                                        self.data.index[self.serial_extrapolated[0][1] + 1].minute,
                                                        self.data.values[self.serial_extrapolated[0][0] - 1],
                                                        self.data.values[self.serial_extrapolated[0][1] + 1]
                                                        )
                        self.balanced_data.values[t] = y
                        # print('after', self.balanced_data.values[t])

                        self.m[t] = 0
                self.data_analysis()
            else:
                break



        # while self.serial_extrapolated[-1][1] - self.serial_extrapolated[-1][0] >\
        #         len(self.balanced_data) - self.serial_extrapolated[-1][1]:
        #
        #     self.balanced_data = self.balanced_data[:self.serial_extrapolated[-1][0] - 1]
        #     self.m = self.m[:self.serial_extrapolated[-1][0] - 1]
        #     self.data_analysis()

        while True:
            if self.serial_extrapolated.__len__() == 0:
                break
            if self.serial_extrapolated[-1][1] - self.serial_extrapolated[-1][0] > len(self.balanced_data) - self.serial_extrapolated[-1][1]:
                if self.change_length==True:
                    self.balanced_data = self.balanced_data[:self.serial_extrapolated[-1][0] - 1]
                    self.m = self.m[:self.serial_extrapolated[-1][0] - 1]
                elif self.change_length==False:
                    # print("Use linear interpolated for keeping length in begin")

                    count = self.serial_extrapolated[-1][1] - self.serial_extrapolated[-1][0] + 1
                    len_between_points = self.balanced_data.index[self.serial_extrapolated[-1][1]].minute - \
                                         self.balanced_data.index[self.serial_extrapolated[-1][0] - 1].minute
                    gamma = len_between_points / count
                    current_minute = self.data.index[self.serial_extrapolated[-1][0] - 1].minute
                    for t in range(self.serial_extrapolated[-1][0], self.serial_extrapolated[-1][1]):
                        current_minute += gamma
                        y = self.__linear_interpolation(current_minute,
                                                        self.data.index[self.serial_extrapolated[-1][0] - 1].minute,
                                                        self.data.index[self.serial_extrapolated[-1][1] + 1].minute,
                                                        self.data.values[self.serial_extrapolated[-1][0] - 1],
                                                        self.data.values[self.serial_extrapolated[-1][1] + 1]
                                                        )
                        self.balanced_data.values[t] = y
                        self.m[t] = 0

                self.data_analysis()
            else:
                break
        if self.serial_extrapolated.__len__()==1:
            # print("linear interpolated for keep length")

            count = self.serial_extrapolated[0][1] - self.serial_extrapolated[0][0] + 1
            len_between_points = self.balanced_data.index[self.serial_extrapolated[0][1]].minute - \
                                 self.balanced_data.index[self.serial_extrapolated[0][0] - 1].minute
            gamma = len_between_points / count
            current_minute = self.data.index[self.serial_extrapolated[0][0] - 1].minute
            for t in range(self.serial_extrapolated[-1][0], self.serial_extrapolated[-1][1]):
                current_minute += gamma
                y = self.__linear_interpolation(current_minute,
                                                self.data.index[self.serial_extrapolated[0][0] - 1].minute,
                                                self.data.index[self.serial_extrapolated[0][1] + 1].minute,
                                                self.data.values[self.serial_extrapolated[0][0] - 2],
                                                self.data.values[self.serial_extrapolated[0][1] + 2]
                                                )
                self.balanced_data.values[t] = y
                self.m[t] = 0
        if self.serial_extrapolated.__len__() > 1:

            last_missing_index = self.serial_extrapolated[0][1]
            distantion = []
            for pair in self.serial_extrapolated[1:]:
                distantion.append(pair[0] - last_missing_index)
                last_missing_index = pair[1]
            argmin_dist = int(np.argmin(distantion))
            for i in range(self.serial_extrapolated[argmin_dist][1], self.serial_extrapolated[argmin_dist + 1][0]):
                self.m[i] = 1
            self.data_analysis()


    # def find_period(self, x):
    #     # вектор ошибок
    #     errors = []
    #     # data = dataset2
    #     values = self.balanced_data.values
    #     # for v in values:
    #     #     print(v)
    #     # print(values.__len__())
    #     alpha, beta, gamma = x
    #
    #     # задаём число фолдов для кросс-валидации
    #     tscv = sklearn.model_selection._split.TimeSeriesSplit(n_splits=3)
    #
    #     # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
    #     Errors = []
    #     for s in range(1, 150):
    #         for train, test in tscv.split(values):
    #
    #             model = HoltWinters(series=values[train], slen=s, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
    #             model.triple_exponential_smoothing()
    #
    #             predictions = model.result[-len(test):]
    #             actual = values[test]
    #             error = mean_squared_error(predictions, actual)
    #             errors.append(error)
    #         Errors.append(np.sum(errors))
    #
    #     # Возвращаем средний квадрат ошибки по вектору ошибок
    #     #     print('s:', s, 'errors:', errors)
    #         errors = []
    #     Errors = np.array(Errors).T
    #     # print(Errors.argsort())


    def timeseriesCVscore(self, x):
        # вектор ошибок
        errors = []

        values = self.balanced_data.values
        alpha, beta, gamma = x
        # задаём число фолдов для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=3)

        # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
        for train, test in tscv.split(values):
            # print('train:', len(train), train)
            # print('test:', len(test), test)

            model = HoltWinters(series=values[train], slen=self.season_len, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
            model.triple_exponential_smoothing()

            test_l = int(len(test) / 3 * 2)
            predictions = model.result[-test_l:]
            # print('predictions len', len(predictions))
            actual = values[test[:test_l]]
            # print('actual len', len(actual))
            error = mean_squared_error(predictions, actual)
            errors.append(error)
        # exit()
        # Возвращаем средний квадрат ошибки по вектору ошибок
        print('коэффициенты:', alpha, beta, gamma, 'errors:', errors)
        return np.mean(np.array(errors))

    def diff_function(self):

        self.diff_balanced_data = pd.DataFrame(data=np.zeros(len(self.balanced_data - 1)), index=self.balanced_data.index[:len(self.balanced_data)])
        for i in range(1, len(self.balanced_data)):
            self.diff_balanced_data.values[i - 1] = self.balanced_data.values[i] - self.balanced_data.values[i - 1]


    def real_data_analysis(self, model):

        self.data.columns = ["Real"]
        self.data["Model"] = np.zeros(len(self.data))
        self.data["UpperBond"] = np.zeros(len(self.data))
        self.data["LowerBond"] = np.zeros(len(self.data))
        index_data = 0
        last_index_balanced_data = 0

        for i in range(1, len(self.balanced_data)):
            if self.balanced_data.values[i] != np.inf:
                while self.data.index[index_data] < self.balanced_data.index[i]:
                    y = self.__linear_interpolation(
                        self.data.index[index_data].minute,
                        self.balanced_data.index[last_index_balanced_data].minute,
                        self.balanced_data.index[i].minute,
                        model.result[last_index_balanced_data],
                        model.result[i]
                                                    )
                    y_upper_bond = self.__linear_interpolation(
                        self.data.index[index_data].minute,
                        self.balanced_data.index[last_index_balanced_data].minute,
                        self.balanced_data.index[i].minute,
                        model.UpperBond[last_index_balanced_data],
                        model.UpperBond[i]
                                                    )
                    y_lower_bond = self.__linear_interpolation(
                        self.data.index[index_data].minute,
                        self.balanced_data.index[last_index_balanced_data].minute,
                        self.balanced_data.index[i].minute,
                        model.LowerBond[last_index_balanced_data],
                        model.LowerBond[i]
                    )


                    self.data["Model"][index_data] = y
                    self.data["UpperBond"][index_data] = y_upper_bond
                    self.data["LowerBond"][index_data] = y_lower_bond

                    index_data += 1
                    if index_data == len(self.data):
                        break
                last_index_balanced_data = i
        self.data["Model"][-1] = self.data["Model"][-2]
        self.data["UpperBond"][-1] = self.data["UpperBond"][-2]
        self.data["LowerBond"][-1] = self.data["LowerBond"][-2]

    def liquidation_missing_values(self):
        self.data_analysis()
        print('discrete_interpolated', self.discrete_interpolated.__len__())
        print('serial_interpolated', self.serial_interpolated.__len__())
        print('serial_extrapolated', self.serial_extrapolated.__len__())
        print('discrete_extrapolated', self.discrete_extrapolated.__len__())

        while self.discrete_interpolated.__len__() > 0 or self.discrete_extrapolated.__len__() > 0:
            self.liquidation_discrete_values()
            self.data_analysis()
        print("discrete values deleted")
        print('discrete_interpolated', self.discrete_interpolated.__len__())
        print('serial_interpolated', self.serial_interpolated.__len__())
        print('serial_extrapolated', self.serial_extrapolated.__len__())
        print('discrete_extrapolated', self.discrete_extrapolated.__len__())
        print()

        while self.serial_interpolated.__len__() > 0:
            # if serial_interpolated_number != TimeSeries.serial_interpolated.__len__():
            # print('less serial interpolated misses')
            #     serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
            self.liquidation_series_interpolated_values()

            self.data_analysis()
            print("inter values deleted")
            print('discrete_interpolated', self.discrete_interpolated.__len__())
            print('serial_interpolated', self.serial_interpolated.__len__())
            print('serial_interpolated', self.serial_interpolated)
            print('serial_extrapolated', self.serial_extrapolated.__len__())
            print('discrete_extrapolated', self.discrete_extrapolated.__len__())
            print()

        self.data_analysis()
        print("inter values deleted")
        print('discrete_interpolated', self.discrete_interpolated.__len__())
        print('serial_interpolated', self.serial_interpolated.__len__())
        print('serial_extrapolated', self.serial_extrapolated.__len__())
        print('discrete_extrapolated', self.discrete_extrapolated.__len__())


        # current_number_extrapolated_series = self.serial_extrapolated.__len__()
        while self.serial_extrapolated.__len__() > 0:
            self.liquidation_series_extrapolated_values()
            self.liquidation_series_interpolated_values()
            self.data_analysis()
            # if current_number_extrapolated_series == TimeSeries.serial_interpolated.__len__():
            #     break
            print("extra deleted")
            print('last discrete_interpolated', self.discrete_interpolated.__len__())
            print('last serial_interpolated', self.serial_interpolated.__len__())
            print('last serial_extrapolated', self.serial_extrapolated.__len__())
            print('last discrete_extrapolated', self.discrete_extrapolated.__len__())
            print(len(self.balanced_data))
            # plotly_df(self.balanced_data, "extra")

            # for i in range(TimeSeries.m.__len__()):
            #     print('after:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i])

        if self.serial_interpolated.__len__() > 0:
            self.liquidation_series_interpolated_values()
        # plotly_df(self.balanced_data, "extra")






