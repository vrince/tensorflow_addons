
"""Image threshold ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_threshold_ops_so = tf.load_op_library(
    resource_loader.get_path_to_datafile("_threshold_ops.so"))

image_threshold = _threshold_ops_so.image_threshold