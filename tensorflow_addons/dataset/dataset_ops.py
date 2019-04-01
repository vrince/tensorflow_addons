
"""Image distance ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import resource_loader
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes

my_reader_dataset_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("_dataset_ops.so"))


class MyReaderDataset(dataset_ops.DatasetSource):

    def __init__(self):
        super(MyReaderDataset, self).__init__(
            my_reader_dataset_module.my_reader_dataset())

    @property
    def _element_structure(self):
        return structure.TensorStructure(dtypes.string, [])
