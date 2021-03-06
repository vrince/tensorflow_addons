from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import threshold
from tensorflow_addons.utils import test_utils

_DTYPES = set([
    tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.float16,
    tf.dtypes.float32, tf.dtypes.float64
])


class ThresholdOpsTest(tf.test.TestCase):
    @test_utils.run_in_graph_and_eager_modes
    def test_simple_threshold(self):
        for dtype in _DTYPES:
            output = threshold.image_threshold(
                [[[1.2, 2.5], [0.2, 4.9]]], [1, 2, 3])
            self.assertAllEqual([[[1, 2], [0, 3]]], output)


if __name__ == "__main__":
    tf.test.main()
