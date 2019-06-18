import tensorflow as tf

# Assumes the file is in the current working directory.
my_reader_dataset_module = tf.load_op_library("./my_reader_dataset_op.so")


class MyReaderDataset(tf.data.Dataset):

    def __init__(self):
        super(MyReaderDataset, self).__init__()
        # Create any input attrs or tensors as members of this class.

    def _as_variant_tensor(self):
        # Actually construct the graph node for the dataset op.
        #
        # This method will be invoked when you create an iterator on this dataset
        # or a dataset derived from it.
        return my_reader_dataset_module.my_reader_dataset()

    # The following properties define the structure of each element: a scalar
    # <a href="../../api_docs/python/tf#string"><code>tf.string</code></a> tensor. Change these properties to match the `output_dtypes()`
    # and `output_shapes()` methods of `MyReaderDataset::Dataset` if you modify
    # the structure of each element.
    @property
    def output_types(self):
        return tf.string

    @property
    def output_shapes(self):
        return tf.TensorShape([])

    @property
    def output_classes(self):
        return tf.Tensor


if __name__ == "__main__":
    # Create a MyReaderDataset and print its elements.
    with tf.Session() as sess:
        iterator = MyReaderDataset().make_one_shot_iterator()
        next_element = iterator.get_next()
        try:
            while True:
                print(sess.run(next_element))  # Prints "MyReader!" ten times.
        except tf.errors.OutOfRangeError:
            pass
