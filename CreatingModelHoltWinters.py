from pickle_data import pickle_it,unpickle_it
from Analysis_TimeSeries import TimeSeriesAnalysis
from data_reader import db_dict_reader
from web_plotly import plotly_df
import numpy as np
from scipy.optimize import minimize
from CVscore import timeseriesCVscore
from HoltWinters import HoltWinters
import matplotlib.pyplot as plt
from XGBoost import XGB_forecast
import pandas as pd
import datetime
import dateparser

def create_HoltWintersModel(data):
    print(data)
    date = data[1]
    date = [dateparser.parse(x) for x in date]
    # date = [datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f") for x in date]
    df = pd.DataFrame(data=data[2], index=date)
    # plotly_df(df)
    number_value_per_hour = int(data[0][0])
    period_day = (df.index[-1] - df.index[0]).days

    TimeSeries = TimeSeriesAnalysis(df, number_value_per_hour, period_day)
    TimeSeries.balancing_data()

    # plotly_df(datafr, 'balanced data with missing values: ')
    # print(TimeSeries.balanced_data)
    # exit()
    # datafr_filter, m = filter_data(datafr, number_value_per_hour, period_day)
    # '''m: измерительная мера'''
    # plotly_df(datafr_filter, 'tag')
    TimeSeries.data_analysis()
    # for i in range(TimeSeries.m.__len__()):
    #     print('before:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i], TimeSeries.marker[i])

    while TimeSeries.discrete_interpolated.__len__() > 0 or TimeSeries.discrete_extrapolated.__len__() > 0:
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        TimeSeries.liquidation_discrete_values()
        TimeSeries.data_analysis()

    # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
    # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
    # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
    # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
    # print()
    # for i in range(TimeSeries.m.__len__()):
    #     print('after:', i, TimeSeries.balanced_data.values[i])
    # plotly_df(TimeSeries.balanced_data, 'balanced data without discrete missing values: ')

    # exit()
    # serial_interpolated_number = TimeSeries.serial_interpolated.__len__() + 1
    while TimeSeries.serial_interpolated.__len__() > 0:
        # if serial_interpolated_number != TimeSeries.serial_interpolated.__len__():
        #     print('less serial interpolated misses')
        #     serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        TimeSeries.liquidation_series_interpolated_values()
        TimeSeries.data_analysis()
        # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        # else:
        #     print("изменений нет")
        #     TimeSeries.liquidation_series_extrapolated_values()
        #     TimeSeries.data_analysis()
        # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        # exit()
        # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()

        # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')

    # TimeSeries.data_analysis()
    # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
    # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
    # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
    # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
    # for i in range(TimeSeries.balanced_data.values.__len__()):
    #     print(i, TimeSeries.balanced_data.values[i])
    # print()

    # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')
    while TimeSeries.serial_extrapolated.__len__() > 0:
        TimeSeries.liquidation_series_extrapolated_values()
        TimeSeries.liquidation_series_interpolated_values()
        TimeSeries.data_analysis()

        # print('last discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('last serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('last serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('last discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())

    plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')
    # инициализируем значения параметров
    x = np.array([0, 0, 0])
    # Минимизируем функцию потерь с ограничениями на параметры
    opt = minimize(TimeSeries.timeseriesCVscore, x0=x, method="TNC", bounds=((0, 1), (0, 1), (0, 1)))
    return opt.x
    # return TimeSeries.balanced_data
    # pickle_it(TimeSeries.balanced_data, 'balanced_data.txt')
    # exit()
    # TimeSeries.balanced_data = unpickle_it('balanced_data.txt')

    # bst = unpickle_it('model_bst.txt')
    # deviation = unpickle_it('deviation_bst.txt')
