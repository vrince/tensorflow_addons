from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.utils.python import test_utils

from tensorflow_addons.dataset import dataset_ops


class DatasetOpsTest(tf.test.TestCase):
    def test_dataset(self):
        dataset = dataset_ops.MyReaderDataset()
        i = 0
        for d in dataset:
            self.assertAllEqual(d, tf.constant("MyReader!"))
            i += 1
        self.assertEquals(i, 10)


if __name__ == "__main__":
    tf.test.main()
