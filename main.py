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
def plotHoltWinters(data, model):

    Anomalies = np.array([np.NaN]*len(data))

    print('Anomalies shape: ', Anomalies.shape)
    # print((data).values)
    # print(model.LowerBond)
    print(len(data.values))
    print(model.LowerBond.__len__())
    print(len([data.values < model.LowerBond]))
    for i in range(len(data.values)):
        if data.values[i] < model.LowerBond[i]:
            print(i)
            Anomalies[i] = data.values[i]

    # print((model.UpperBond))
    # print(np.shape(model.LowerBond))

    # exit()
    for i in range(len(data.values)):
        if data.values[i] > model.UpperBond[i]:
            print(i)
            Anomalies[i] = data.values[i]
    # exit()
    # Anomalies[data.values < model.LowerBond] = data.values[data.values < model.LowerBond]
    # # exit()
    # Anomalies[data.values > model.UpperBond] = data.values[data.values > model.UpperBond]
    # print('Anomalies2: ', Anomalies)
    plt.figure(figsize=(15, 6))

    print(len(model.result))

    plt.plot(model.result, label="Model")
    plt.plot(model.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
    plt.plot(model.LowerBond, "r--", alpha=0.5)
    plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color="grey")
    plt.plot(data.values, label="Actual", c='green')
    plt.plot(Anomalies, "o", markersize=6, label="Anomalies", c='red')
    plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13)
    plt.show()


if __name__=='__main__':


    # for i in range(5, 100):
    # 6,8,16,18,19,24,25,28,32,36,41,44,47,50
    # datafr = db_dict_reader(18)
    # print(datafr)
    # pickle_it(datafr,'data18.txt')
    # exit()
    datafr = unpickle_it('data18.txt')
    # print(datafr)
    # exit()
    # plotly_df(datafr, 'Original data')

    number_value_per_hour = 4
    period_day = 50
    TimeSeries = TimeSeriesAnalysis(datafr, number_value_per_hour, period_day)
    TimeSeries.balancing_data()

    # plotly_df( TimeSeries.balanced_data, 'balanced data with missing values: ')
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


    print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
    print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
    print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
    print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
    print()
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
        print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
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

    # plotly_df(TimeSeries.balanced_data, 'balanced data without of series interpolated missing values: ')



    TimeSeries.data_analysis()
    print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
    print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
    print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
    print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
    for i in range(TimeSeries.balanced_data.values.__len__()):
        print(i, TimeSeries.balanced_data.values[i])
    print()

    # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')
    while TimeSeries.serial_extrapolated.__len__() > 0:
        TimeSeries.liquidation_series_extrapolated_values()
        TimeSeries.liquidation_series_interpolated_values()
        TimeSeries.data_analysis()
        print('last discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        print('last serial_interpolated', TimeSeries.serial_interpolated.__len__())
        print('last serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        print('last discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
    # plotly_df(TimeSeries.balanced_data, 'balanced data without of missing values: ')

    # pickle_it(TimeSeries.balanced_data, 'balanced_data.txt')
    # exit()
    # TimeSeries.balanced_data = unpickle_it('balanced_data.txt')


    # bst = unpickle_it('model_bst.txt')
    # deviation = unpickle_it('deviation_bst.txt')




    # XGB_forecast(TimeSeries.balanced_data, test_size=0.2)
    # exit()
    # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')





    # from pandas import Series
    # from matplotlib import pyplot
    # from statsmodels.graphics.tsaplots import plot_acf

    # series = TimeSeries.balanced_data.values
    # plot_acf(series)
    # pyplot.show()
    # exit()

    # инициализируем значения параметров
    x = np.array([0, 0, 0])
    # Минимизируем функцию потерь с ограничениями на параметры
    # opt = minimize(TimeSeries.timeseriesCVscore, x0=x, method="TNC", bounds=((0, 1), (0, 1), (0, 1)))
    #
    # print(opt)
    #
    # # Из оптимизатора берем оптимальное значение параметров
    # alpha_final, beta_final, gamma_final = opt.x
    # print(alpha_final, beta_final, gamma_final)
    alpha_final, beta_final, gamma_final = [0.03080408, 0.01077995, 0.19563582]
    # pickle_it([alpha_final,beta_final,gamma_final], 'coef.txt')
    # exit()
    # alpha_final, beta_final, gamma_final = unpickle_it('coef.txt')
    # print(alpha_final, beta_final, gamma_final)
    # Передаем оптимальные значения модели,
    # data_ = dataset2
    model = HoltWinters(TimeSeries.balanced_data.values, slen=8, alpha=alpha_final, beta=beta_final, gamma=gamma_final, n_preds=0,
                        scaling_factor=3.7)
    model.triple_exponential_smoothing()
    #
    # pickle_it(model, "model_holt_winters.txt")
    # model = unpickle_it("model_holt_winters.txt")
    # print(model.UpperBond)

    plotHoltWinters(TimeSeries.balanced_data, model)
