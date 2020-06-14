# import time
import zmq
from web_plotly import plotly_df2
# import numpy as np
from CreatingModelHoltWinters import HoltWintersModel
import enum
# import matplotlib.pyplot as plt
# import dateparser
# import pandas as pd
from XGBoost import XGBoost_create_model, XGBoost_prediction
# from pickle_data import unpickle_it
# import  json
# " SELECT DBTimeStamp, Value FROM SEICDatabase.dbo.SEICEVENTHISTORY WHERE Ev_Info = @param and DBTimeStamp > @start and DBTimeStamp < @end"
# class TimeSeriesData:
#     def __init__(self, points, series_type):
#         self.Points = points
#         self.SeriesType = series_type
# pyinstaller main.py -F --add-data "venv/Lib/site-packages/xgboost/VERSION;xgboost/" --add-data "venv/Lib/site-packages/xgboost/xgboost.dll;xgboost/lib/" --onefile --add-data "venv/Lib/site-packages/sklearn/.libs/vcomp140.dll;."
# S01N00145U001N01D0035N01PAI____PI00   S01N00413U001N01D0035N01PAI____PI00  'alpha': 0.11857221342498864, 'betta': 0.0148639144477688, 'gamma': 2.7881349839997682e-05, 'season_len': 2}
class SeriesType(enum.Enum):
        BuilderHoltWinters = 0
        BuilderXGBoost = 1
        ForecastingHoltWinters = 2
        ForecastingXGBoost = 3

# def plot(data):
#     plt.figure(figsize=(15, 6))
#     a = data["Real"]
#     # print()
#
#     date = [dateparser.parse(x["Date"]) for x in a]
#     # print(date)
#     values = [x["Value"] for x in a]
#     # print(values)
#
#     indices = np.array(date).argsort()
#     dict_a = dict()
#     for i in indices:
#         dict_a.setdefault(date[i], values[i])
#     # print(dict_a.values())
#     df = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())
#
#     plt.plot(df.index, df.values, label="Model")
#     plt.plot(model.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
#     plt.plot(model.LowerBond, "r--", alpha=0.5)
#     plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color="grey")
#     plt.plot(data.values, label="Actual", c='green')
#     plt.plot(Anomalies, "o", markersize=6, label="Anomalies", c='red')
#     plt.axvspan(len(data) - 128, len(data), alpha=0.5, color='lightgrey')
#     plt.grid(True)
#     plt.axis('tight')
#     plt.legend(loc="best", fontsize=13)
#     plt.show()


def create_connection():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")
    data = []

    while True:
        #  Wait for next request from client
        print('Wait for next request from client')

        # holt = unpickle_it("rmse_holt.txt")
        # xg = unpickle_it("rmse_xgb.txt")
        # plt.figure(figsize=(15, 15))
        # plt.title("RMSE")
        # plt.plot(holt, label="HoltWinters")
        # plt.plot(xg, label="XGBoost")
        # plt.grid(True)
        # plt.axis('tight')
        # plt.legend(loc="best", fontsize=13)
        # plt.show()
        data = socket.recv_json()

        result = []
        if data["SeriesType"] == SeriesType.BuilderHoltWinters.value:
            HWModel = HoltWintersModel(data)
            result = HWModel.create_HoltWintersModel()
            # result =  create_HoltWintersModel(data)
            # print(result)
            # data = []
            socket.send_json(result)
        elif data["SeriesType"] == SeriesType.ForecastingHoltWinters.value:
            HWModel = HoltWintersModel(data)
            result = HWModel.analysis_timeseries_Holt_Winters()
            # print((result))
            try:
                json_string = result.to_json(orient='index', date_format='iso')
            except:
                json_string = ""
            # print(json_string)
            socket.send_string(json_string)
            if data["PlotBrawser"]==True:
                plotly_df2(result, "HoltWinters Anomaly Detection" + data["Data"]["Name"])
            # plot(result[])
        elif data["SeriesType"] == SeriesType.BuilderXGBoost.value:
            result = XGBoost_create_model(data)
            print(result)
            # json_string = result.to_json()
            # print(json_string)
            # result = json.dumps(result, default=lambda o: o.dict)
            socket.send_json(result)
        elif data["SeriesType"] == SeriesType.ForecastingXGBoost.value:
            print(data)
            result = XGBoost_prediction(data, data["ScalingFactor"])
            print((result))
            # json_string = result.to_json(orient='index', date_format='iso')
            # print(json_string)
            json_string = result.to_json(orient='index', date_format='iso')
            socket.send_string(json_string)
            if data["PlotBrawser"]==True:
                plotly_df2(result, "XGBoost Anomaly Detection" + data["Data"]["Name"])








        # elif data["SeriesType"] == SeriesType.ForecastingHoltWinters:
            # socket.send_json("111")
            # x = socket.recv_json()
            # print(data)
            # print()
            # print(x)
            # result = analysis_timeseries_Holt_Winters(data, x)
        #  Do some 'work'
        # time.sleep(1)
        #  Send reply back to client
        #     socket.send_json(result)