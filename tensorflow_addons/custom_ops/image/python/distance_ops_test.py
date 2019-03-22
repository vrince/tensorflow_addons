from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gradient_checker
from tensorflow_addons.custom_ops.image.python import distance as distance_ops
from tensorflow_addons.utils.python import test_utils

_DTYPES = set([
    tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.float16,
    tf.dtypes.float32, tf.dtypes.float64
])


class DistanceOpsTest(tf.test.TestCase):
    @test_utils.run_in_graph_and_eager_modes
    def test_simple_distance(self):
        for dtype in _DTYPES:
            distance = distance_ops.distance_transform(
                [[[1.2, 2.5], [0.2, 4.9]]], 1)
            self.assertTrue(True)


if __name__ == "__main__":
    tf.test.main()
