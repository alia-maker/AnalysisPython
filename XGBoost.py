import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from pickle_data import pickle_it, unpickle_it
import pandas as pd
# from sklearn.metrics import mean_absolute_error
def PrepareData(data, test_size=0.15):
    # data = pd.DataFrame(data.copy()),
    # data.columns = ["y"]
       # считаем индекс в датафрейме, после которого начинается тестовыый отрезок"
    test_index = int(len(data)*(1-test_size))
    # data.columns = ['y']
    # data = data.reset_index(drop=True)
    # data.astype(int)
    #
    # print(data)
    # exit()
    # Data = []
    # for d in data:
    #     Data.extend(d.tolist())

    # "    # добавляем лаги исходного ряда в качестве признаков"
    # for i in range(lag_start, lag_end):
    #     data["lag_{}".format(i)] = data.y.shift(i)
    #
    #     data.index = data.index.to_datetime()
    #     data["hour"] = data.index.hour
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
    # X_train = data.loc[:test_index].drop(["y"], axis=1)
    X_train =(data[:test_index])
    print(X_train)
    # y_train = data.loc[:test_index]["y"]
    y_train =(data.values[:test_index])
    print(y_train)
    # X_test = data.loc[test_index:].drop(["y"], axis=1)
    # y_test = data.loc[test_index:]["y"]
    X_test = (data[test_index + 20:])
    y_test = (data.values[test_index + 20:])

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


def XGB_forecast(data, test_size=0.15, lag_start=5, lag_end=30, scale=1.9):
    print('real data:', data)
    # исходные данные
    X_train, X_test, y_train, y_test = PrepareData(data)
    print('X train', X_train)
    print('y train', y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    # exit()
    dtest = xgb.DMatrix(X_test)

    # задаём параметры
    params = {
        'objective': 'reg:linear',
        'booster':'gblinear'
    }
    trees = 300

    # прогоняем на кросс-валидации с метрикой rmse
    cv = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=False, nfold=10, show_stdv=False, num_boost_round=trees)

    # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
    bst = xgb.train(params, dtrain, num_boost_round=cv['test-rmse-mean'].argmin())

    # можно построить кривые валидации
    #cv.plot(y=['test-mae-mean', 'train-mae-mean'])

    # запоминаем ошибку на кросс-валидации
    deviation = cv.loc[cv['test-rmse-mean'].argmin()]["test-rmse-mean"]
    pickle_it(bst, 'model_bst.txt')
    pickle_it(deviation, 'deviation_bst. txt')
    # exit()
    # посмотрим, как модель вела себя на тренировочном отрезке ряда
    prediction_train = bst.predict(dtrain)
    plt.figure(figsize=(15, 5))
    # print(len(prediction_train))
    plt.plot(y_train.index, prediction_train)
    # print(y_train)
    # exit()
    plt.plot(y_train)
    plt.axis('tight')
    plt.grid(True)
    # plt.show()
    # exit()
    # и на тестовом
    prediction_test = bst.predict(dtest)
    lower = prediction_test-scale*deviation
    upper = prediction_test+scale*deviation

    Anomalies = np.array([np.NaN]*len(y_test))
    # print(y_test)
    # print(lower)
    # exit()
    for i in range(len(y_test)):
        if y_test.values[i] < lower[i]:

            Anomalies[i] = y_test.values[i]
    for i in range(len(y_test)):
        if y_test.values[i] > upper[i]:

            Anomalies[i] = y_test.values[i]
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.index, prediction_test, label="prediction")
    plt.plot(y_test.index, lower, "r--", label="upper/lower bound")
    plt.plot(y_test.index, upper, "r--")
    plt.plot(y_test, label="y_test")
    print(Anomalies)
    # exit()
    plt.plot(y_test.index, Anomalies, "ro", markersize=10, label="Anomalies")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("XGBoost")

    # plt.title("XGBoost Mean absolute error {} users".format(round(mean_absolute_error(prediction_test, y_test))))
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__=='__main__':
    dataset = pd.read_csv('hour_online.csv', index_col=['Time'], parse_dates=['Users'])
    # dataset.astype(dtype=int)
    # data = pd.DataFrame(np.arange(12).reshape((4, 3)), columns=['a', 'b', 'c'])
    # label = pd.DataFrame(np.random.randint(2, size=4))
    # print(data)
    # print(label)
    # dtrain = xgb.DMatrix(data, label=label)
    # print(dtrain)
    # exit()


    XGB_forecast(dataset, test_size=0.2, lag_start=5, lag_end=30)
