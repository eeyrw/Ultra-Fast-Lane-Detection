from tensorflow.core.util import event_pb2
import tensorflow as tf
from collections import defaultdict, namedtuple
from typing import List
event_filename = 'Ultra-Fast-Lane-Detection\events.out.tfevents.1606619470.gpu01.38411.0'
# for e in tf.compat.v1.train.summary_iterator('Ultra-Fast-Lane-Detection\events.out.tfevents.1606619470.gpu01.38411.0'):
#     b = e.summary
#     for v in b.value:
#         if v.value.tag == 'loss_relation_loss/relation_loss_6_S3':
#             print(v)

infoDict = defaultdict(list)
serialized_examples = tf.data.TFRecordDataset(event_filename)
for serialized_example in serialized_examples:
    event = event_pb2.Event.FromString(serialized_example.numpy())
    for value in event.summary.value:
        #t = tf.make_ndarray(value.tensor)
        # print(value.tag, event.step, value.simple_value)
        infoDict[value.tag].append((event.step, value.simple_value))

a = 2
