
"""Image distance ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_distance_ops_so = tf.load_op_library(
    get_path_to_datafile("custom_ops/image/_distance_ops.so"))

distance_transform_3d = _distance_ops_so.distance_transform3d
distance_transform_2d = _distance_ops_so.distance_transform2d
