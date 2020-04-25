import time
import zmq
import numpy as np
from CreatingModelHoltWinters import create_HoltWintersModel

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
data = []

while True:
    #  Wait for next request from client
    x = socket.recv_json()
    # print("Received request: ", np.array(x))
    data.append(x)
    # y = socket.recv_json()
    # print("Received request: ", np.array(y).astype('double'))
    if len(data) == 3:
        result = create_HoltWintersModel(data)
        print(result)
        data = []
    #  Do some 'work'
    # time.sleep(1)

    #  Send reply back to client
        socket.send_json(result)