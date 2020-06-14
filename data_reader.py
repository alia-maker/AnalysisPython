# import csv
import numpy as np
import pyodbc
# import pickle
from pickle_data import pickle_it,unpickle_it
import pandas as pd
# import datetime



def db_dict_reader(ind):
    # ev_info = []
    # s1 = []
    # s2 = []
    # s3 = []
    #
    # t1 = []
    # t2 = []
    # t3 = []
    driver = 'DRIVER={SQL Server}'
    server = 'SERVER=DESKTOP-SSLIKJH'
    port = 'PORT=1433'
    db = 'DATABASE=SEICDatabase'
    user = 'Aliya Sultanova'
    pw = 'PWD='
    conn_str = ';'.join([driver, server, port, db, user, pw])

    conn = pyodbc.connect(conn_str)
    # with conn:
    #     print("connection done")
        # cursor = conn.cursor()
    #     cursor.execute( '''
    #                     select DISTINCT Ev_Info FROM SEICDatabase.dbo.SEICEVENTHISTORY;
    #                     ''')
    #     row = cursor.fetchone()
    #     print("ev info got")
    #     while row:
    #         # if ev_info.count(row[0])==0:
    #         ev_info.append(row[0])
    #             # print(row[0])
    #         row = cursor.fetchone()
    # print(ev_info.__len__())
    # pickle_it(ev_info, 'ev_info.txt')
    ev_info = unpickle_it("ev_info.txt")
    DataTime = []
    Value = []
    dict_a = dict()
    with conn:
        print('The connection to the database is successful')
        cursor = conn.cursor()
        cursor.execute('''
                        SELECT DBTimeStamp, Value FROM SEICDatabase.dbo.SEICEVENTHISTORY WHERE Ev_Info=?;''', (ev_info[ind]))
        row = cursor.fetchone()

        while row:
            DataTime.append((row[0]))
            Value.append(float(row[1]))
            row = cursor.fetchone()

    indices = np.array(DataTime).argsort()
    print(ind, ": ", len(indices))
    for i in indices:
        dict_a.setdefault(DataTime[i], Value[i])

    print(dict_a.values())
    # exit()
    df = pd.DataFrame(data=dict_a.values(), index=dict_a.keys())
    print(df.values.tolist())
    # df = pd.DataFrame(data=Value, index=DataTime)

    return df

# from plotly import graph_objs as go
# from plotly.offline import plot
#
#
#
#
# def plotly_df(df, title = ''):
#     data = []
#
#     for column in df.columns:
#         trace = go.Scatter(
#             x=df.index,
#             y=df[column],
#             mode='lines',
#             name=column
#         )
#         data.append(trace)
#
#     layout = dict(title=title)
#     fig = dict(data=data, layout=layout)
#     plot(fig, show_link=False)

# def filter_data(data_, number_value, period):
#     m = []
#     '''m: измерительная мера'''
#     DateTime_new = []
#     # first_time_hour = data_.index[0].hour
#     # first_time_minute = data_.index[0].minute
#     # print(first_time_hour, first_time_minute)
#     interval_value = np.round(np.linspace(0, 60, number_value), 0).astype(int)
#     max_dif = int((interval_value[1]-interval_value[0])) * 60 / 3 * 2
#
#     dif_interval = np.abs(interval_value - data_.index[0].minute)
#     argmin_value = np.argmin(dif_interval)
#
#     first_time = datetime.datetime(year=data_.index[0].year, month=data_.index[0].month, day=data_.index[0].day)
#
#     if interval_value[argmin_value] < 60:
#         a = datetime.timedelta(hours=data_.index[0].hour, minutes=int(interval_value[argmin_value]))
#     else:
#         a = datetime.timedelta(hours=data_.index[0].hour + 1, minutes=0)
#
#     first_time = first_time.__add__(a)
#     DateTime_new.append(first_time)
#
#
#     temp_time = first_time
#     while temp_time < first_time.__add__(datetime.timedelta(days=period)):
#         temp_time = temp_time.__add__(datetime.timedelta(minutes=int(60 / number_value)))
#         DateTime_new.append(temp_time)
#     i = 0
#     Value_new = []
#     for timestamp in DateTime_new:
#         similar_values = []
#         for k in range(i, i+20):
#             similar_values.append(data_.index[k])
#
#         dif_values = []
#         for v in similar_values:
#             dif_values.append(abs(v - timestamp))
#         argmin_v = int(np.argmin(dif_values))
#         # print('timestamp:', timestamp, 'min dif seconds:', dif_values[argmin_v].seconds)
#         if dif_values[argmin_v].days == 0 and dif_values[argmin_v].seconds < max_dif:
#             Value_new.append(float(data_.values[i + argmin_v]))
#             m.append(0)
#         else:
#             Value_new.append(0)
#             m.append(2)
#         i += argmin_v
#
#     print(Value_new.count(0))
#     # print(Value_new)
#     print(DateTime_new.__len__())
#
#     df = pd.DataFrame(data=Value_new, index=DateTime_new)
#     return df, m

import enum
class Status(enum.Enum):
    obs = 0
    miss = 1


# class Mark(enum.Enum):
#     discrete_interpolated = 1
#     serial_interpolated = 2
#     serial_extrapolated = 3
#     discrete_extrapolated = 4
#
# def data_analysis(m, data):
#     marker = np.zeros(m.__len__())
#     discrete_interpolated = []
#     serial_interpolated = []
#     serial_extrapolated = []
#     discrete_extrapolated = []
#     state = Status.miss.value
#     i_obs = 0
#     i_miss = 0
#
#     for i in range(len(m)):
#         if Status.miss.value == state:
#             if m[i] > 0:
#                 i_miss += 1
#             else:
#                 state = Status.obs.value
#
#         if Status.obs.value == state:
#             if m[i] == 0:
#                 i_obs += 1
#             else:
#
#                 state = Status.miss.value
#                 if i_miss == 1 and i_obs > i_miss:
#                     discrete_interpolated.append(i - 1 - i_obs)
#                     # print('discrete_interpolated++ index:', i - i_obs - 1)
#                     marker[i - i_obs - 1] = Mark.discrete_interpolated.value
#                 elif i_miss == 1 and i_obs <= i_miss:
#                     discrete_extrapolated.append(i - 1 - i_obs)
#                     marker[i - 1 - i_obs] = Mark.discrete_extrapolated.value
#
#                 elif i_miss > 1 and i_obs > i_miss:
#                     # serial_interpolated += 1
#                     marker[i - i_miss - i_obs:i-i_obs] = Mark.serial_interpolated.value
#                     serial_interpolated.append([i - i_miss - i_obs, i - i_obs])
#                 elif i_miss > 1 and i_obs <= i_miss:
#                     serial_extrapolated.append([i - i_miss - i_obs, i - i_obs])
#                     marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_extrapolated.value
#                     # i_miss = 0
#                     # i_obs = 0
#                 i_miss = 1
#                 i_obs = 0
#         print(data.values[i], m[i])
#         print('obs', i_obs, 'miss', i_miss)
#
#     i = len(m)
#     if Status.obs.value == state:
#         if i_miss == 1 and i_obs > i_miss:
#             discrete_interpolated.append(i - 1 - i_obs)
#             print('discrete_interpolated++ index:', i - i_obs - 1)
#             marker[i - i_obs - 1] = Mark.discrete_interpolated.value
#         elif i_miss == 1 and i_obs <= i_miss:
#             discrete_extrapolated.append(i - 1 - i_obs)
#             marker[i - i_obs] = Mark.discrete_extrapolated.value
#
#         elif i_miss > 1 and i_obs > i_miss:
#             serial_interpolated.append([i - i_miss - i_obs, i - i_obs])
#             marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_interpolated.value
#         elif i_miss > 1 and i_obs <= i_miss:
#             serial_extrapolated.append([i - i_miss - i_obs, i - i_obs])
#             marker[i - i_miss - i_obs:i - i_obs] = Mark.serial_extrapolated.value
#
#     # for t in range(m.__len__()):
#     #     print(data.values[t], m[t], marker[t])
#     # print('discrete_interpolated:', discrete_interpolated)
#     # print('serial_interpolated:', serial_interpolated)
#     # print('serial_extrapolated:', serial_extrapolated)
#     # print('discrete_extrapolated', discrete_extrapolated)
#     return discrete_interpolated, discrete_extrapolated, serial_interpolated, serial_extrapolated
#

def linear_interpolation(data, timestamps):
    # print(timestamps)
    # print(data.values[timestamps[0] - 1], data.values[timestamps[0] + 1])
    x1 = data.index[timestamps[0] - 1].minute
    x2 = data.index[timestamps[0] + 1].minute
    if x1 > x2:
        x1 -= 60
    y1 = data.values[timestamps[0] - 1]
    y2 = data.values[timestamps[0] + 1]
    print(x1, x2, y1, y2)
    x = data.index[timestamps[0]].minute
    if x > x2:
        x -= 60

    a = (y2 - y1)/(x2 - x1)
    b = y1 - a * x1
    y = a * x + b
    print(y)

    pass

if __name__=='__main__':
    # for i in range(5, 100):
    # 6,8,16,18,19,24,25,28,32,36,41,44,47,50
    # datafr = db_dict_reader(18)
    # print(datafr)
    # pickle_it(datafr,'data18.txt')
    # exit()
    datafr = unpickle_it('data18.txt')
    # plotly_df(datafr, 'Tag: ')
    TimeSeries = TimeSeriesAnalysis(datafr, )
    number_value_per_hour = 4
    period_day = 30
    datafr_filter, m = filter_data(datafr, number_value_per_hour, period_day)
    '''m: измерительная мера'''
    # plotly_df(datafr_filter, 'tag')

    discrete_interpolated, discrete_extrapolated, serial_interpolated, serial_extrapolated = data_analysis(m, datafr_filter)

    linear_interpolation(datafr_filter, discrete_interpolated)



    # for i in range(0, 1000, 10):
    #     print(datafr[i:i+10])
    # print('plotly:')
    #

