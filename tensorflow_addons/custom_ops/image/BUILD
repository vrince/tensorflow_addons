licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")

cc_binary(
    name = "python/_distance_ops.so",
    srcs = glob(["cc/kernels/distance_transform/**",
        "cc/kernels/distance_op.cpp",
        "cc/kernels/distance_op.h",
        "cc/ops/distance_ops.cc",
    ]),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

cc_binary(
    name = "python/_threshold_ops.so",
    srcs = [
        "cc/kernels/image_threshold_op.cc",
        "cc/kernels/image_threshold_op.h",
        "cc/ops/threshold_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

cc_binary(
    name = "_distort_image_ops.so",
    srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.cc",
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/ops/distort_image_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        D_GLIBCXX_USE_CXX11_ABI,
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_image_ops.so",
    srcs = [
        "cc/kernels/euclidean_distance_transform_op.cc",
        "cc/kernels/euclidean_distance_transform_op.h",
        "cc/kernels/image_projective_transform_op.cc",
        "cc/kernels/image_projective_transform_op.h",
        "cc/ops/image_ops.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        D_GLIBCXX_USE_CXX11_ABI,
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "images_ops_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/distance.py", 
        "python/threshold.py",
        "python/distort_image_ops.py",
        "python/transform.py",
    ]),
    data = [
        ":python/_distance_ops.so",
        ":python/_threshold_ops.so",
        ":python/_distort_image_ops.so",
        ":python/_image_ops.so",
        "//tensorflow_addons/utils:utils_py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "distance_ops_test",
    size = "small",
    srcs = [
        "python/distance_ops_test.py",
    ],
    main = "python/distance_ops_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3"
)

py_test(
    name = "threshold_ops_test",
    size = "medium",
    srcs = [
        "python/threshold_ops_test.py",
    ],
    main = "python/threshold_ops_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3"
)


py_test(
    name = "distort_image_ops_test",
    size = "small",
    srcs = [
        "python/distort_image_ops_test.py",
    ],
    main = "python/distort_image_ops_test.py",
    deps = [
        ":images_ops_py",
    ],
    srcs_version = "PY2AND3"
)

# TODO: use cuda_py_test later.
py_test(
    name = "transform_ops_test",
    size = "medium",
    srcs = [
        "python/transform_test.py",
    ],
    main = "python/transform_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":images_ops_py",
    ],
)
