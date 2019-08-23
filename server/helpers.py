import io
import zlib

import numpy as np
import cv2

def norm_img(img):
    
    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    return img.astype(np.float32)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return



def compress_np(nparr):
    """Compress numpy array to bytestream."""
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    compressed = zlib.compress(bytestream.getvalue())
    return bytestream

def decompress_np(bytestring):
    """Decompress numpy array from bytestream."""
    return np.load(io.BytesIO(zlib.decompress(bytestring)))