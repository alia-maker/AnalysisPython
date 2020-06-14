# from pickle_data import pickle_it,unpickle_it
from Analysis_TimeSeries import TimeSeriesAnalysis
# from data_reader import db_dict_reader
# from web_plotly import plotly_df, plotly_df2
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize
# from CVscore import timeseriesCVscore
from HoltWinters import HoltWinters
# import matplotlib.pyplot as plt
# from XGBoost import XGB_forecast
import pandas as pd
# import datetime
import dateparser
# from UsingHoltWintersExample import plotHoltWinters
from scipy.signal import find_peaks
# from statsmodels.graphics import tsaplots
from collections import Counter
# import sklearn
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


class HoltWintersModel:
    def __init__(self, data):
        self.data = data

    def read_data(self):
        a = self.data["Points"]
        date = [dateparser.parse(x["Date"]) for x in a]
        values = [x["Value"] for x in a]
        indices = np.array(date).argsort()
        dict_a = dict()
        for i in indices:
            dict_a.setdefault(date[i], values[i])
        self.df = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())


    def create_HoltWintersModel(self):
        try:
            self.read_data()
            print("Данные успешно обработаны")
        except Exception:
            print("Полученные данные имеют неверную структуру")
            return {
                    "model_created": False,
                    "alpha": int(0),
                    "betta": int(0),
                    "gamma": int(0),
                    "season_len": int(0),
                }

        try:
            number_value_per_hour = self.data["NumberOfValue"]
            period_day = (self.df.index[-1] - self.df.index[0]).days
            TimeSeries = TimeSeriesAnalysis(self.df, number_value_per_hour, period_day, change_length=True)
            TimeSeries.balancing_data()
            TimeSeries.liquidation_missing_values()
            for i in range(len(TimeSeries.balanced_data)):
                print(TimeSeries.balanced_data.values[i])
        except Exception:
            print("Возникла ошибка при запонении пропусков")
            return {
                "model_created": False,
                "alpha": int(0),
                "betta": int(0),
                "gamma": int(0),
                "season_len": int(0),
                "PlotBrawser":False,
            }

        try:
            if TimeSeries.serial_interpolated.__len__() == 0 and TimeSeries.serial_extrapolated.__len__() == 0 and len(TimeSeries.balanced_data) > 200:
                # TimeSeries.diff_function()
                v = [float(x) for x in TimeSeries.balanced_data.values]
                l = int(v.__len__() / 3)
                acf2 = acf(v[:250], nlags=250, fft=True)
                acf1 = (acf2)
                print('acf1', acf1)

                t = find_peaks(acf1)[0]
                print('peaks', t)

                diff = []
                if t.__len__() < 2:
                    TimeSeries.season_len = 10
                else:
                    for i in range(1, len(t)):
                        diff.append(t[i] - t[i-1])
                    print('periods:', diff)
                    c = Counter(diff).most_common(1)
                    TimeSeries.season_len = c[0][0]
                x = np.array([0, 0, 0]).astype(float)
                # Минимизируем функцию потерь с ограничениями на параметры
                opt = minimize(TimeSeries.timeseriesCVscore, x0=x, method="TNC", bounds=((0, 1), (0, 1), (0, 1)))
                return {
                    "model_created": True,
                    "alpha": (opt.x[0]),
                    "betta": (opt.x[1]),
                    "gamma": (opt.x[2]),
                    "season_len": int(TimeSeries.season_len),
                    "PlotBrawser": True,
                }
            else:
                print("Модель не построена. Мало данных")
                return {
                    "model_created": False,
                    "alpha": int(0),
                    "betta": int(0),
                    "gamma": int(0),
                    "season_len": int(0),
                    "PlotBrawser": False,

                }
        except Exception:
            "Ошибка построения модели Хольта-Уинтерса"
            return {
                "model_created": False,
                "alpha": int(0),
                "betta": int(0),
                "gamma": int(0),
                "season_len": int(0),
                "PlotBrawser":False,

            }


    def read_analys_data(self):
        points = self.data["Data"]["Points"]
        print("points")
        date = [dateparser.parse(x["Date"]) for x in points]
        print("date")
        values = [x["Value"] for x in points]
        print("values")
        indices = np.array(date).argsort()
        dict_a = dict()
        for i in indices:
            dict_a.setdefault(date[i], values[i])
        self.df = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())
        print((self.df))
    def analysis_timeseries_Holt_Winters(self):
        try:
            self.read_analys_data()
            print("Входные данные успешно обработаны")
        except Exception:
            print("Входные данные имеют неверную структуру")
            return pd.DataFrame()
        try:

            period_day = (self.df.index[-1] - self.df.index[0]).days
            print('period', period_day)
            number_value_per_hour = int(len(self.df) / (period_day * 24))
            if number_value_per_hour > 1:
                number_value_per_hour -= 1
            print("number_value_per_hour", number_value_per_hour)
            TS = TimeSeriesAnalysis(self.df, number_value_per_hour, period_day, change_length=False)
            TS.balancing_data()
            print(TS.balanced_data)
            print("Создан временной ряд")
            TS.liquidation_missing_values()
            print("Заполнены пропуски")
            print("длина ряда:", len(TS.balanced_data))
            for i in range(len(TS.balanced_data)):
                print(TS.balanced_data.values[i])
        except Exception:
            print("Преобразование данных не выполнено")
            return pd.DataFrame()
        try:
            acf2 = acf(TS.balanced_data.values[:250], nlags=250, fft=True)
            acf1 = (acf2)
            print('acf1', acf1)
            t = find_peaks(acf1)[0]
            print('peaks', t)
            diff = []
            if t.__len__() < 2:
                season_len = 16
            else:
                for i in range(1, len(t)):
                    diff.append(t[i] - t[i - 1])
                print('periods:', diff)
                c = Counter(diff).most_common(1)
                season_len = c[0][0]
            season_len *= 3
        except Exception:
            print("Ошибка при вычислении длины сезона")
            return pd.DataFrame()

        try:
            model = HoltWinters(TS.balanced_data.values, slen=season_len, alpha=self.data["alpha"], beta=self.data["betta"],
                                gamma=self.data["gamma"], n_preds=0,
                                scaling_factor=self.data["ScalingFactor"])
            model.triple_exponential_smoothing()
            print("Анализ ВР проведен")

            TS.real_data_analysis(model)
            print("Получен результат")

            # if True:
            #     print("Рисунок в браузере")
            TS.data["PlotBrawser"]  = True
            return TS.data
        except Exception:
            print("При анализе произошли ошибки")
            return pd.DataFrame()