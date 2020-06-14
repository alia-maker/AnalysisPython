# import numpy as np
# import matplotlib.pyplot as plt
from ServerZeroMQ import create_connection

# def plotHoltWinters(data, model):
#     Anomalies = np.array([np.NaN]*len(data))
#     print('Anomalies shape: ', Anomalies.shape)
#     print(len(data.values))
#     print(model.LowerBond.__len__())
#     print(len([data.values < model.LowerBond]))
#     for i in range(len(data.values)):
#         if data.values[i] < model.LowerBond[i]:
#             print(i)
#             Anomalies[i] = data.values[i]
#
#     for i in range(len(data.values)):
#         if data.values[i] > model.UpperBond[i]:
#             print(i)
#             Anomalies[i] = data.values[i]
#     plt.figure(figsize=(15, 6))
#     print(len(model.result))
#     plt.plot(model.result, label="Model")
#     plt.plot(model.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
#     plt.plot(model.LowerBond, "r--", alpha=0.5)
#     plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color="grey")
#     plt.plot(data.values, label="Actual", c='green')
#     plt.plot(Anomalies, "o", markersize=6, label="Anomalies", c='red')
#     plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')
#     plt.grid(True)
#     plt.axis('tight')
#     plt.legend(loc="best", fontsize=13)
#     plt.show()

if __name__=='__main__':
    create_connection()

