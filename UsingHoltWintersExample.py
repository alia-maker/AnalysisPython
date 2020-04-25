
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs
from plotly import graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.offline
from HoltWinters import HoltWinters
from scipy.optimize import minimize
import numpy as np
from sklearn.metrics import mean_squared_error


def plotly_df(df, title = ''):
    data = []

    for column in df.columns:
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        )
        data.append(trace)

    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # прогнозируем
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotMovingDoubleExponential(dataset):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20, 8))
        for alpha in [0.9, 0.02]:
            for beta in [0.9, 0.02]:
                plt.plot(double_exponential_smoothing(dataset.values.astype(float), alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(dataset.values.astype(float), label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()
from sklearn.model_selection import TimeSeriesSplit

def timeseriesCVscore(x):
    # вектор ошибок
    errors = []
    # data = dataset2
    values = data.values
    print(values)
    alpha, beta, gamma = x

    # задаём число фолдов для кросс-валидации
    tscv = TimeSeriesSplit(n_splits=3)

    # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], slen = 400, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)

    # Возвращаем средний квадрат ошибки по вектору ошибок
    return np.mean(np.array(errors))

def plotHoltWinters(data, model):

    Anomalies = np.array([np.NaN]*len(data))
    print('Anomalies shape: ', Anomalies.shape)
    # print((data).values)
    # print(model.LowerBond)
    Anomalies[data.values < model.LowerBond] = data.values[data.values < model.LowerBond]
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

if __name__ == "__main__":
    # print("Begin of program")
    dataset = pd.read_csv('daily-minimum-temperatures-in-me.csv', index_col=['Date'], parse_dates=['Daily minimum temperatures'])
    print(type(dataset))
    print((dataset.axes))
    # dataset.set
    # exit()
    # plt.plot([1,2,3],[2,5,3])
    # plt.show()
    data = dataset[:400]
    # print(dataset2)
    #
    # plotly_df(data, title="Online users")
    exit()





    # plotMovingDoubleExponential(dataset)# сглаживание двойное экспоненциальное

    ##### holt_winters = HoltWinters(dataset, 24, 0.5, 0.5, 0.5, 24, 2.5)

    # % % time
    # data = dataset.Users[:-500]  # отложим часть данных для тестирования

    # инициализируем значения параметров
    x = np.array([0, 0, 0])
    # print(type(x))
    # exit()
    # Минимизируем функцию потерь с ограничениями на параметры
    opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds=((0, 1), (0, 1), (0, 1)))

    # Из оптимизатора берем оптимальное значение параметров
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)

    # Передаем оптимальные значения модели,
    # data_ = dataset2
    model = HoltWinters(data, slen=400, alpha=alpha_final, beta=beta_final, gamma=gamma_final, n_preds=128,
                        scaling_factor=2.56)
    model.triple_exponential_smoothing()
    #
    # pickle_it(model, "model_holt_winters.txt")
    # model = unpickle_it("model_holt_winters.txt")
    # print(model.UpperBond)

    plotHoltWinters(data, model)


