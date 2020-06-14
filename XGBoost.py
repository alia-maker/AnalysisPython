import xgboost as xgb
import numpy as np
import os

from pickle_data import pickle_it, unpickle_it
import pandas as pd
import dateparser
from Analysis_TimeSeries import TimeSeriesAnalysis
from web_plotly import plotly_df2
from UsingHoltWintersExample import plotHoltWinters
import pickle
import json
import base64
import datetime
import sklearn
# from sklearn.metrics import mean_absolute_error
def PrepareData(data, test_size, start_lag, end_lag):
    # data = pd.DataFrame(data.copy()),
    # data.columns = ["y"]
       # считаем индекс в датафрейме, после которого начинается тестовыый отрезок"
    test_index = int(len(data)*(1-test_size))
    data.columns = ['y']
    # data = data.reset_index(drop=True)
    # data.astype(int)

    # print(data)

    # data['date'] = data.index
    # print(data)
    # data['hour'] = data['date'].dt.hour
    # print(data)
    # df['dayofweek'] = df['date'].dt.dayofweek
    # df['quarter'] = df['date'].dt.quarter
    # df['month'] = df['date'].dt.month
    # df['year'] = df['date'].dt.year
    # df['dayofyear'] = df['date'].dt.dayofyear
    # df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear

    # Data = []
    # for d in data:
    #     data.extend(d.tolist())

    "    # добавляем лаги исходного ряда в качестве признаков"
    for i in range(start_lag+4, end_lag*5, 5):
        temp_lag = data.y.shift(i)
        # print(type(temp_lag[0]))
        temp_lag[:i] = 0
        # print(temp_lag)
        # exit()
        data["lag_{}".format(i)] = temp_lag
    print(data)
    # print(type(data.index[0]))
    # exit()
    # print(data.index)

    # data.index = pd.to_datetime(str(date))

    # data.index = [dateparser.parse(x) for x in data.index]
    # exit()
    data["minute"] = data.index.minute
    data["hour"] = data.index.hour
    data['quarter'] = data.index.quarter
    data["weekday"] = data.index.weekday
    # print(data)
    # exit()
    # "    data[\"weekday\"] = data.index.weekday\n",
    # "    data['is_weekend'] = data.weekday.isin([5,6])*1\n",
    # "\n",
    # "    # считаем средние только по тренировочной части, чтобы избежать лика\n",
    # "    data['weekday_average'] = map(code_mean(data[:test_index], 'weekday', \"y\").get, data.weekday)\n",
    # "    data[\"hour_average\"] = map(code_mean(data[:test_index], 'hour', \"y\").get, data.hour)\n",
    # "\n",
    # "    # выкидываем закодированные средними признаки \n",
    # "    data.drop([\"hour\", \"weekday\"], axis=1, inplace=True)\n",
    # "\n",
    # "    data = data.dropna()\n",
    # "    data = data.reset_index(drop=True)\n",
    # "\n",
    # разбиваем весь датасет на тренировочную и тестовую выборку\n",
    # data = data.reset_index(drop=True)
    # X_train = data[:test_index].drop(["y"], axis=1)
    # print(X_train)
    print(data)
    print((data))

    # X_train = data[:test_index].drop(["y"], axis=1)
    X_train = data[:test_index].drop(["y"], axis=1)
    print('X_train', X_train)
    # X_test = data[test_index:].drop(["y"], axis=1)
    X_test = data[test_index:].drop(["y"], axis=1)
    print('X_test', X_test)

    # y_train = data[:test_index]["y"]
    y_train = data[:test_index]["y"]
    # y_test = data[test_index:]["y"]
    y_test = data[test_index:]["y"]
    # print(y_train)
    # print(y_test)
    # exit()
    # X_test = data.loc[test_index:].drop(["y"], axis=1)
    # y_test = data.loc[test_index:]["y"]
    # X_test = (data[test_index + 20:])
    # y_test = (data.values[test_index + 20:])

    return X_train, X_test, y_train, y_test



def code_mean(data, cat_feature, real_feature):
    """
    Возвращает словарь, где ключами являются уникальные категории признака cat_feature,
    а значениями - средние по real_feature
    """
    # print(type(data))
    # data = pd.to_numeric(data)
    # print((data.groupby('weekday')['y'].mean()))
    # exit()
    return dict(data.groupby(cat_feature)[real_feature].transform("mean"))

def prepareData(data, lag_start=5, lag_end=20, test_size=0.15):
    data = pd.DataFrame(data.copy())
    data.columns = ["y"]
    "    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок"
    test_index = int(len(data)*(1-test_size))
    "    # добавляем лаги исходного ряда в качестве признаков"
    # for i in range(lag_start, lag_end):
    #     data["lag_{}".format(i)] = data.y.shift(i)
    # print(data.index)
    # data.index = pd.to_datetime(data.index)
    # data["hour"] = data.index.hour
    # data["weekday"] = data.index.weekday
    # data['is_weekend'] = data.weekday.isin([5, 6])*1
    # print(data)
        # exit()
    # считаем средние только по тренировочной части, чтобы избежать лика

    # exit()
    # data['weekday_average'] = data.groupby['weekday'].mean()
    # print(data)
    # exit()
    # data['weekday_average'] = map(code_mean(data[:test_index], 'weekday', 'y').get, data.weekday)
    # data["hour_average"] = map(code_mean(data[:test_index], 'hour', "y").get, data.hour)
    # выкидываем закодированные средними признаки
    # data.drop(["hour", "weekday"], axis=1, inplace=True)
    # data = data.dropna()
    data = data.reset_index(drop=True)
    # разбиваем весь датасет на тренировочную и тестовую выборку
    print(data)
    # exit()
    X_train = data[:test_index].drop(["y"], axis=1)
    # exit()
    y_train = data[:test_index]["y"]
    X_test = data[test_index:].drop(["y"], axis=1)
    y_test = data[test_index:]["y"]
    return X_train, X_test, y_train, y_test


def XGB_forecast(data, test_size, lag_start=5, lag_end=30, oname = ""):
    print('real data:', data)
    # исходные данные
    X_train, X_test, y_train, y_test = PrepareData(data, test_size, lag_start, lag_end)
    # print('X train', np.array(X_train.values).astype(float))
    # print('y train', y_train)
    # exit()

    X = np.array(X_train.values).astype(float)
    y = np.array(y_train.values).astype(float)
    dtrain = xgb.DMatrix(X, label=y)
    # exit()
    # XTest = np.array(X_test.values).astype(float)
    # ytest = np.array(y_test.values).astype(float)
    # dtest = xgb.DMatrix(XTest)

    # задаём параметры
    params = {
        'booster': 'gblinear',
        'objective': 'reg:squarederror'
    }
    trees = 2000

    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=True, nfold=10, show_stdv=True, num_boost_round=trees)
    print('cv:', cv)
    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    # plt.figure(figsize=(15, 15))
    print(cv['test-rmse-mean'].argmin())
    # plt.title("RMSE")
    # plt.plot(cv['train-rmse-mean'], label="train rmse", c='green')
    #
    # plt.plot(cv['test-rmse-mean'],label="test rmse", c='red')
    # plt.grid(True)
    # plt.axis('tight')
    # plt.legend(loc="best", fontsize=13)
    # plt.xlabel = "Number of trees"
    # plt.ylabel = "Error"
    # plt.show()
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())
    # print(bst)
    # config = bst.save_config()
    # print(config)

    # file_name = oname + '_bst_model_' + str(datetime.datetime.today().date()) + '_' + \
    #             str(datetime.datetime.today().hour) + ':' + str(datetime.datetime.today().minute) + '.json'

    # Create target Directory if don't exist
    dirName = "PythonData"
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    file_name = oname + '_bst_model_' + str(datetime.datetime.today().date()) + '_' + \
                str(datetime.datetime.today().hour) + ':' + str(datetime.datetime.today().minute) + '.json'
    path = os.path.join('PythonData', file_name)
    print(path)
    bst.save_model(path)

    # pickle_it()
    # with open(oname + '.json') as f:
    #     bst_json = json.load(f)
    # print(type(bst_json))
    # print('bst json', bst_json)
    # bst_pickle = pickle.dumps(bst)
    # print('pickle:')
    # print(bst_pickle)
    # string_bst_pickle = bst_pickle.decode(encoding='unicode_escape')


    # string_bst_pickle = base64.b64decode(bst_pickle)

    # print(type(string_bst_pickle))
    # print((string_bst_pickle))
    # print('encode pickle:', base64.b64encode(string_bst_pickle))
    # bst = pickle.loads(base64.b64encode(string_bst_pickle))


    result = {
        "Deviation": cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"],
        "ModelPath": file_name
    }
    return result



def XGBoost_prediction(data, scale):
    try:
        # print(data)
        deviation = data["Deviation"]
        path = os.path.join('PythonData', data["ModelPath"])
        bst = xgb.Booster(model_file=path)
        print('bst', bst)
        points = data["Data"]["Points"]
        print('points')
        date = [dateparser.parse(x["Date"]) for x in points]
        print('date',date)
        values = [x["Value"] for x in points]
        print('values', values)
        indices = np.array(date).argsort()
        dict_a = dict()
        for i in indices:
            dict_a.setdefault(date[i], values[i])

        test_data = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())
        print(test_data)
        # for i in range(len(test_data.values) - 1070, len(test_data.values) - 840):
        #     test_data.values[i] = 11.2
    except:
        print("Неверная структура входных данных")
        return 0
    print("Входные данные обработаны успешно")
    # with open(data["ModelPath"]) as f:
    #     bst.load_model(f)
    # print(data["Model"])
    # print(type(data["Model"]))
    # print()
    # print(data["Model"].encode('unicode_escape'))
    # print(type(data["Model"].encode('unicode_escape')))

    # bst = pickle.loads(data["Model"].encode('unicode_escape'))

    # bst = pickle.loads(base64.b64encode(data["Model"]))
    # print(bst)
    # test_data = data["Data"]
    try:
        result = test_data.copy()
        X_train, X_test, y_train, y_test = PrepareData(test_data, test_size=1,start_lag=1, end_lag=30)
        XTest = np.array(X_test.values).astype(float)
        ytest = np.array(y_test.values).astype(float)
        dtest = xgb.DMatrix(XTest)
        # xgb.plot_importance(bst)
        # xgb.plot_importance(bst, height=0.9)

        # можно построить кривые валидации
        #cv.plot(y=['test-mae-mean', 'train-mae-mean'])

        # запоминаем ошибку на кросс-валидации
        # deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]
        # pickle_it(bst, 'model_bst.txt')
        # pickle_it(deviation, 'deviation_bst. txt')
        # exit()
        # посмотрим, как модель вела себя на тренировочном отрезке ряда
        # prediction_train = bst.predict(dtrain)


        # plt.figure(figsize=(15, 5))
        # print(len(prediction_train))
        # print(y_train)
        # print(y_train.index)
        # print(len(y_train.index))
        # print(prediction_train)
        # print(len(prediction_train))
        # exit()
        # plt.plot(y_train.index, prediction_train)
        # print(y_train)
        # exit()
        # plt.plot(y_train.index, y)
        # plt.axis('tight')
        # plt.grid(True)
        # plt.show()
        # exit()
        # и на тестовом

        prediction_test = bst.predict(dtest)
        lower = prediction_test-scale*deviation
        upper = prediction_test+scale*deviation

        # for i in range(len(test_data.values) - 1070, len(test_data.values) - 840):
        #     lower[i] +=1
        #     upper[i]-=1.4
        # print('resulr:', result)
        result.columns = ["Real"]
        result["Model"] = prediction_test
        result["UpperBond"] = upper
        result["LowerBond"] = lower
        # plotHoltWinters(result)
        # print('test data:', test_data)
        rmse_xg = []
        # for i in range(910, len(result) - 10):
        #     w = sklearn.metrics.mean_squared_error(result["Real"][900:i],result["Model"][900:i])
        #     rmse_xg.append(w)
        # pickle_it(rmse_xg, "rmse_xgb.txt")
        # if True:
        #     plotly_df2(result, "XGBoost Anomaly Detection" + data["Data"]["Name"])
        return result
    except Exception:
        print("При анализе данных произошли ошибки")
        return 0


def XGBoost_create_model(data):
    print(data)
    try:
        a = data["Points"]
        object_name = str(data["Name"])
        # print(a)
        # print(type(a[0]["Date"]))
        date = [dateparser.parse(x["Date"]) for x in a]
        # print(date)
        values = [x["Value"] for x in a]
        # print(values)
        # values[100:150] += 4
        indices = np.array(date).argsort()
        dict_a = dict()
        for i in indices:
            dict_a.setdefault(date[i], values[i])
        # print(dict_a.values())
        df = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())

        # plotly_df(df, "Реальная последовательность значений")
    except Exception:
        print('Неверная структура входных данных')
        return 0
    print("Входные данные обработаны успешно")
        # print(df)
    try:
        number_value_per_hour = data["NumberOfValue"]
        # print("number_value_per_hour", number_value_per_hour)
        period_day = (df.index[-1] - df.index[0]).days
        # print(period_day)
        TimeSeries = TimeSeriesAnalysis(df, number_value_per_hour, period_day, change_length=True)
        TimeSeries.balancing_data()
        TimeSeries.liquidation_missing_values()
        for i in range(len(TimeSeries.balanced_data)):
            print(TimeSeries.balanced_data.values[i])
        # print(TimeSeries.balanced_data)

        # datafr_filter, m = filter_data(datafr, number_value_per_hour, period_day)
        # '''m: измерительная мера'''
        # plotly_df(datafr_filter, 'tag')
        # TimeSeries.data_analysis()
        # # for i in range(TimeSeries.m.__len__()):
        # #     print('before:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i], TimeSeries.marker[i])
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        # while TimeSeries.discrete_interpolated.__len__() > 0 or TimeSeries.discrete_extrapolated.__len__() > 0:
        #     # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        #     # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        #     # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        #     # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        #     TimeSeries.liquidation_discrete_values()
        #     TimeSeries.data_analysis()
        # # print(len(TimeSeries.balanced_data))
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        # print()
        # # for i in range(TimeSeries.m.__len__()):
        # #     print('after:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i])
        # # plotly_df(TimeSeries.balanced_data, 'balanced data without discrete missing values: ')
        #
        # # exit()
        # # serial_interpolated_number = TimeSeries.serial_interpolated.__len__() + 1
        # while TimeSeries.serial_interpolated.__len__() > 0:
        #     # if serial_interpolated_number != TimeSeries.serial_interpolated.__len__():
        #     #     print('less serial interpolated misses')
        #     #     serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        #     TimeSeries.liquidation_series_interpolated_values()
        #     TimeSeries.data_analysis()
        #     # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        #     # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        #     # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        #     # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        #     # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        #     # else:
        #     #     print("изменений нет")
        #     #     TimeSeries.liquidation_series_extrapolated_values()
        #     #     TimeSeries.data_analysis()
        #     # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        #     # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        #     # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        #     # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        #     # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        #     # exit()
        #     # serial_interpolated_number = TimeSeries.serial_interpolated.__len__()
        #
        # plotly_df(TimeSeries.balanced_data, 'balanced data without interpolated series of missing values: ')
        #
        # TimeSeries.data_analysis()
        # print('discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        # print('serial_interpolated', TimeSeries.serial_interpolated.__len__())
        # print('serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        # print('discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        #
        #
        # # for i in range(TimeSeries.m.__len__()):
        # #     print('after:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i])
        # # for i in range(TimeSeries.balanced_data.values.__len__()):
        # #     print(i, TimeSeries.balanced_data.values[i])
        # # print()
        # # print(len(TimeSeries.balanced_data))
        #
        # # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')
        # current_number_extrapolated_series = TimeSeries.serial_extrapolated.__len__()
        # while TimeSeries.serial_extrapolated.__len__() > 0:
        #     TimeSeries.liquidation_series_extrapolated_values()
        #     TimeSeries.liquidation_series_interpolated_values()
        #     TimeSeries.data_analysis()
        #     # if current_number_extrapolated_series == TimeSeries.serial_interpolated.__len__():
        #     #     break
        #
        #
        #
        #     print('last discrete_interpolated', TimeSeries.discrete_interpolated.__len__())
        #     print('last serial_interpolated', TimeSeries.serial_interpolated.__len__())
        #     print('last serial_extrapolated', TimeSeries.serial_extrapolated.__len__())
        #     print('last discrete_extrapolated', TimeSeries.discrete_extrapolated.__len__())
        #     print(len(TimeSeries.balanced_data))
        #     # for i in range(TimeSeries.m.__len__()):
        #     #     print('after:', i, TimeSeries.balanced_data.values[i], TimeSeries.m[i])
        #
        # if TimeSeries.serial_interpolated.__len__() > 0:
        #     TimeSeries.liquidation_series_interpolated_values()
        # plotly_df(TimeSeries.balanced_data)
        # result = {
        #     "model_created": False,
        #     "alpha": 0,
        #     "betta": 0,
        #     "gamma": 0,
        #     "season_len": 1,
        # }
    except Exception:
        print("Заполнены не пропуски")
        return 0
    print("Заполнены восстанавливаемые пропуски")
    try:
        if TimeSeries.serial_interpolated.__len__() == 0 and TimeSeries.serial_extrapolated.__len__() == 0 and len(
                TimeSeries.balanced_data) > 200:
            # plotly_df(TimeSeries.balanced_data, 'balanced data without part of missing values: ')
            # инициализируем значения параметров
            # if len(TimeSeries.balanced_data) < 2:
            #     TimeSeries.balanced_data = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())

            # print('number nan', list(values).count(np.inf), list(values).count(np.nan))
            # print('v', list(v))
            # acf1 = autocorr(v)
            # x = [int(t) for t in range(0, len(acf1))]

            TimeSeries.diff_function()
            # plotly_df(TimeSeries.balanced_data)
            # plotly_df(TimeSeries.diff_balanced_data, 'diff_ts')
            model = XGB_forecast(TimeSeries.balanced_data, test_size=0, lag_start=1, lag_end=30, oname=object_name)
            model["Name"] = object_name
            model["PlotBrawser"] = True
            return model
        else:
            return  {
            "cv": "",
            "bst": "",
            "Name": "",
            "PlotBrawser":False,
        }
    except Exception:
        print("При построении модели произошли ошибки")


# if __name__=='__main__':
#     dataset = pd.read_csv('hour_online.csv', index_col=['Time'], parse_dates=['Users'])
#     # dataset.astype(dtype=int)
#     # data = pd.DataFrame(np.arange(12).reshape((4, 3)), columns=['a', 'b', 'c'])
#     # label = pd.DataFrame(np.random.randint(2, size=4))
#     # print(data)
#     # print(label)
#     # dtrain = xgb.DMatrix(data, label=label)
#     # print(dtrain)
#     # exit()
#
#     XGB_forecast(dataset, test_size=0.2, lag_start=5, lag_end=30)
