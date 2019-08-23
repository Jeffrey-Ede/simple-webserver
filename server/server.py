from flask import Flask, request
from flask_socketio import SocketIO, emit, send

from socketIO_client import SocketIO as client_SocketIO

from scipy import ndimage as nd

import numpy as np
import cv2
import zlib

from helpers import *

#from magnifier_net import magnifier_net, test_fn

import time


app = Flask(__name__)
app.config['SECRET_KEY'] = 'NeuralNetworkServer'
socketio = SocketIO(app, async_mode="gevent", engineio_logger=True)

client_socketIO = None

cropsize = 512
def make_mask(use_frac):

    d = int(np.sqrt(1/use_frac))

    mask = np.zeros((cropsize, cropsize))
    for x in range((cropsize%d)//2, cropsize, d):
        for y in range((cropsize%d)//2, cropsize, d):
            mask[y,x] = 1.

    return mask

def fill(data, invalid):
    return data[tuple(nd.distance_transform_edt(invalid, return_distances=False, return_indices=True))]

mask = make_mask(1/25)

invalid = np.logical_not(mask.astype(np.bool))

@socketio.on('connect')
def connect():
    global client_socketIO
    client_socketIO = client_SocketIO('137.205.164.177', 8001, async_mode="gevent")

@socketio.on("process")
def dlss(stringarray): #stringarray

    print("Arrived at server")
    #outputs = np.random.rand(512,512)
    #string = zlib.compress(outputs.astype(np.float16).tostring()).hex()
    
    client_socketIO.emit("processed", stringarray)
    return

    t0 = time.process_time()

    inputs = np.fromstring(zlib.decompress(bytes.fromhex(stringarray)), dtype=np.float16).reshape((103,103)).astype(np.float32)

    lq = cv2.resize(inputs, (515, 515), interpolation=cv2.INTER_NEAREST)[1:-2, 1:-2]

    #outputs = pool.apply_async(magnifier_net, (lq,)).get()

    #outputs = np.expand_dims(np.expand_dims(lq**2 + lq - lq**3, 0), -1)
    #outputs = magnifier_net(lq)

    outputs = np.random.rand(512,512)
    #outputs = np.expand_dims(np.expand_dims(cv2.resize(lq, (512, 512)), 0), -1)

    #emit("processed", {"string": zlib.compress((128*norm_img(outputs)).clip(-127, 128).astype(np.int8).tostring()).hex()})
    string = zlib.compress(outputs.astype(np.float16).tostring()).hex()

    print(time.process_time()-t0)

    emit("processed", {"data": string})#, room=request.sid

#@socketio.on("connect", namespace='/dlss')
#def connect():
#    send("Client connected")

#@socketio.on("disconnect", namespace='/dlss')
#def disconnect():
#    send("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8000)