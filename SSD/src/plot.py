##
## MIT License
## 
## Copyright (c) 2017 Luca Angioloni
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
import argparse
import colorsys
import imghdr
import os
import random
import logging
import math
import datetime
import pickle

import numpy as np
import matplotlib as mpl 
#mpl.use('Qt5Agg')

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

data_frames = pickle.load( open( "data_frames.p", "rb" ) )
objs = {}
# just some random data 
i = 0

colors = np.random.rand(100)

# event listener 
def press(event):
    global i
    if i < len(data_frames):
        ax.cla()
        if event.key == '1':
    		# If press 1 do something different
            print(i)
        #points = []
        points = [obj['coord'] for obj in data_frames[i]]
        ids = [obj['id'] for obj in data_frames[i]]
        i += 1
        #print(points)
        for j in range(len(ids)):
            id = ids[j]
            if id in objs:
                objs[id].append(points[j])
            else:
                objs[id] = [points[j]]
        for o in objs:
            ax.plot([p[0] for p in objs[o]], [p[1] for p in objs[o]])
        fig.canvas.draw()

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
#imgplot = ax.plot(frames[i % 100]) #initial frame
plt.show()