from tensorflow.core.util import event_pb2
import tensorflow as tf
from collections import defaultdict, namedtuple
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

logs = [
    {'name': 'tusimple_semi_0.05',
        'path': './experiment_log/semi_tusimple_finished_20iter_0.05'},
    {'name': 'tusimple_semi_0.1',
        'path': './experiment_log/semi_tusimple_finished_20iter_0.1'},
    {'name': 'tusimple_semi_0.2',
        'path': './experiment_log/semi_tusimple_finished_20iter_0.2'},
    # {'name': 'culane_train_0.5_semi_0.05',
    #          'path': './experiment_log/semi_culane_finished_train_0.5_20iter_0.05'}
]


def readExprLog(logDictList):
    logGroupDict = {}
    for logInfo in logDictList:
        name = logInfo['name']
        path = logInfo['path']
        infoDict = defaultdict(list)
        serialized_examples = tf.data.TFRecordDataset(path)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
                infoDict[value.tag].append((event.step, value.simple_value))
        logGroupDict[name] = dict(infoDict)
    return logGroupDict


def getLogDataWithMultiStep(logGroupDict, group, cat, name, iter=0, step=0):
    return logGroupDict[group]['%s_%s/%s_%d_S%d' % (cat, name, name, iter, step)]


def getLogDataSummary(logGroupDict, group, cat, name):
    return logGroupDict[group]['%s_summary/%s' % (cat, name)]


def make_plot(dataList, labelList, iterEpoch, xlabel, ylabel, title):
    markerList = ['x', 'o', 's']
    fig, ax = plt.subplots(figsize=(4, 2.7), constrained_layout=True)

    for data, label, marker in zip(dataList, labelList, markerList):
        data = np.array(data)
        ax.plot(data[:, 0], data[:, 1], label=label, marker=marker, alpha=0.7)

        # Annotate the 1st position with a text box ('Test 1')
        offsetbox = TextArea('%.3f' % data[0, 1], minimumdescent=False)

        ab = AnnotationBbox(offsetbox, (data[0, 0], data[0, 1]),
                            xybox=(30, 15),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))

        ax.add_artist(ab)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if iterEpoch is not None:
        iterEpoch = np.array(iterEpoch)

        def iter2Epoch(x):
            return np.interp(x, iterEpoch[:, 0], iterEpoch[:, 1])

        def epoch2Iter(x):
            return np.interp(x, iterEpoch[:, 1], iterEpoch[:, 0])

        secax = ax.secondary_xaxis('top', functions=(iter2Epoch, epoch2Iter))
        secax.set_xlabel('Epoch')
    plt.savefig(title+'.pdf')
    plt.show()


logGroupDict = readExprLog(logs)


dataList = [
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.05', 'test', 'Accuracy'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.1', 'test', 'Accuracy'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.2', 'test', 'Accuracy')
]

labalList = [
    '0.05 of full dataset',
    '0.1 of full dataset',
    '0.2 of full dataset'
]

dataList2 = [
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.05', 'test', 'FN'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.1', 'test', 'FN'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.2', 'test', 'FN')
]

dataList3 = [
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.05', 'test', 'FP'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.1', 'test', 'FP'),
    getLogDataSummary(logGroupDict, 'tusimple_semi_0.2', 'test', 'FP')
]

# iterEpoch = getLogDataWithMultiStep(
#     logGroupDict, 'tusimple_semi_0.05', 'meta', 'epoch', 4, 2)

make_plot(dataList, labalList, None,
          'Iteration', 'Accuracy', 'Accuracy')
make_plot(dataList2, labalList, None,
          'Iteration', 'FN', 'FN')
make_plot(dataList3, labalList, None,
          'Iteration', 'FP', 'FP')

