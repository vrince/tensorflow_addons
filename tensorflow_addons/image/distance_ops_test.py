from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import distance as distance_ops
from tensorflow_addons.utils import test_utils

_DTYPES = set([
    tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.float16,
    tf.dtypes.float32, tf.dtypes.float64
])


class DistanceOpsTest(tf.test.TestCase):
    @test_utils.run_in_graph_and_eager_modes
    def test_distance_3d(self):
        distance = distance_ops.distance_transform_3d(
            [[[1., 2., 2.],
              [2., 2., 2.],
              [2., 2., 2.]],
             [[2., 2., 2.],
              [2., 2., 2.],
              [2., 2., 2.]],
             [[2., 2., 2.],
              [2., 2., 2.],
              [2., 2., 2.]]],
            1.5)
        self.assertAllClose(distance, [[[0., 1., 2.],
                                        [1., 1.4142135, 2.236068],
                                        [2., 2.236068, 2.828427]],
                                       [[1., 1.4142135, 2.236068],
                                        [1.4142135, 1.7320508, 2.4494898],
                                        [2.236068, 2.4494898, 3.]],
                                       [[2., 2.236068, 2.828427],
                                        [2.236068, 2.4494898, 3.],
                                        [2.828427, 3., 3.4641016]]])

    @test_utils.run_in_graph_and_eager_modes
    def test_distance_2d(self):
        distance = distance_ops.distance_transform_2d(
            [
                [1., 2., 2.],
                [2., 2., 2.],
                [2., 2., 2.]
            ], 1.5)
        self.assertAllClose(distance, [[0., 1., 2.],
                                       [1., 1.4142135, 2.236068],
                                       [2., 2.236068, 2.828427]])

    @test_utils.run_in_graph_and_eager_modes
    def test_squared_distance_2d(self):
        distance = distance_ops.distance_transform_2d(
            [
                [1., 2., 2.],
                [2., 2., 2.],
                [2., 2., 2.]
            ], 1.5,
            squared=True)
        self.assertAllClose(distance, [[0., 1., 4.],
                                       [1., 2, 5],
                                       [4., 5, 8]])


if __name__ == "__main__":
    tf.test.main()
