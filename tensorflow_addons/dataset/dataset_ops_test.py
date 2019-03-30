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
        i = iter(dataset)
        #next_element = next(i)
        # print(next_element)


if __name__ == "__main__":
    tf.test.main()
