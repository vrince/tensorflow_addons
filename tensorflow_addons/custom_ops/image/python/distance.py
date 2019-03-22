
"""Image distance ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_distance_ops_so = tf.load_op_library(
    resource_loader.get_path_to_datafile("_distance_ops.so"))

distance_transform = _distance_ops_so.distance_transform
