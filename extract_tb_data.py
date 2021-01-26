from tensorflow.core.util import event_pb2
import tensorflow as tf
from collections import defaultdict, namedtuple
from typing import List
import matplotlib.pyplot as plt
import numpy as np
event_filename = 'Ultra-Fast-Lane-Detection\events.out.tfevents.1606619470.gpu01.38411.0'


def getID(cat, name, iter=0, step=0):
    return '%s_%s/%s_%d_S%d' % (cat, name, name, iter, step)


def make_plot(data, epoch):
    fig, ax = plt.subplots(constrained_layout=True)
    data = np.array(data)
    epoch = np.array(epoch)
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel('Iteration Times')
    ax.set_ylabel('Accuracy')
    ax.set_title('Iter 0 Step 0')

    def deg2rad(x):
        return np.interp(x, epoch[:, 0], epoch[:, 1])

    def rad2deg(x):
        return np.interp(x, epoch[:, 1], epoch[:, 0])

    secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
    secax.set_xlabel('Epoch')
    plt.show()


infoDict = defaultdict(list)
serialized_examples = tf.data.TFRecordDataset(event_filename)
for serialized_example in serialized_examples:
    event = event_pb2.Event.FromString(serialized_example.numpy())
    for value in event.summary.value:
        #t = tf.make_ndarray(value.tensor)
        # print(value.tag, event.step, value.simple_value)
        infoDict[value.tag].append((event.step, value.simple_value))
infoDict = dict(infoDict)

make_plot(infoDict[getID('loss', 'aux_loss', 4, 2)],
          infoDict[getID('meta', 'epoch', 4, 2)])
