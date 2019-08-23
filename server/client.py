from socketIO_client import SocketIO
import zlib
import numpy as np

from scipy.misc import imread

import cv2

from helpers import *

model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/stem-random-walk-nin-20-54/"
addr = model_dir + "truth-975000.tif"
inputs = imread(addr, mode='F')

inputs = np.random.rand(512,512)
#inputs = cv2.resize(inputs, (103,103), interpolation=cv2.INTER_AREA)

#maximum = np.max(inputs)
#minimum = np.min(inputs)

#inputs = (128*norm_img(inputs)).clip(-127, 128)

###############################################################################

message = zlib.compress(inputs.astype(np.float16).tostring()).hex()

import time
t0 = time.time()

def handle_processed_data(stringarray):
    
    #img = np.fromstring(zlib.decompress(bytes.fromhex(stringarray["data"])), dtype=np.float16).reshape((512,512))
    print("time", time.process_time()-t0)
    
    #disp(img)

#def handle_processed_data(stringarray):
    
#    #img = np.fromstring(zlib.decompress(bytes.fromhex(stringarray)), dtype=np.float16).reshape((512,512))
#    print("time", time.process_time()-t0)
    
#    #disp(img)

#Establish connection to server
socketIO = SocketIO('137.205.164.177', 8000, async_mode="gevent", engineio_logger=True)
#socketIO = SocketIO('137.205.164.200', 8000, async_mode="gevent", engineio_logger=True)
socketIO.wait(seconds=1)

socketIO.on("processed", handle_processed_data)

###############################################################################

for _ in range(5):
    t0 = time.process_time()
    #socketIO.emit('process', message)
    socketIO.emit('process', message)
    
    socketIO.wait(seconds=1)
    #socketIO.wait(seconds=1)
