
"""Image distance ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import resource_loader
from tensorflow.python.data.ops import dataset_ops

my_reader_dataset_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("_dataset_ops.so"))


class MyReaderDataset(dataset_ops.DatasetSource):

    def __init__(self):
        super(MyReaderDataset, self).__init__(
            my_reader_dataset_module.my_reader_dataset())
        # Create any input attrs or tensors as members of this class.

    # def _as_variant_tensor(self):
    #    # Actually construct the graph node for the dataset op.
    #    #
    #    # This method will be invoked when you create an iterator on this dataset
    #    # or a dataset derived from it.
    #    return my_reader_dataset_module.my_reader_dataset()

    def _element_structure(self):
        structure = DatasetStructure(structure_lib.convert_legacy_structure(
            self.output_types, self.output_shapes, self.output_classes))
        return structure

    @property
    def output_types(self):
        return tf.string

    @property
    def output_shapes(self):
        return tf.TensorShape([])

    @property
    def output_classes(self):
        return tf.Tensor
