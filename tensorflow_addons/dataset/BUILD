licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "_dataset_ops.so",
    srcs = [
        "cc/my_dataset.cpp"
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    # see why -DNDEBUG https://github.com/tensorflow/tensorflow/issues/17316
    copts = ["-pthread","-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DNDEBUG"]
)

py_library(
    name = "dataset",
    srcs = ([
        "__init__.py",
        "dataset_ops.py",
    ]),
    data = [
        ":_dataset_ops.so",
        "//tensorflow_addons/utils",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "dataset_ops_test",
    size = "small",
    srcs = [
        "dataset_ops_test.py",
    ],
    main = "dataset_ops_test.py",
    deps = [
        ":dataset",
    ],
    srcs_version = "PY2AND3"
)
