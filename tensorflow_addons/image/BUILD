licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "image",
    srcs = ([
        "__init__.py",
        "distance.py", 
        "threshold.py",
        "dense_image_warp.py",
        "distance_transform.py",
        "distort_image_ops.py",
        "filters.py",
        "transform_ops.py",
        "translate_ops.py",
        "utils.py",
        "sparse_image_warp.py",
        "interpolate_spline.py",
        "connected_components.py",
        "resampler_ops.py",
    ]),
    data = [
        ":sparse_image_warp_test_data",
        "//tensorflow_addons/custom_ops/image:_distort_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_image_ops.so",
        "//tensorflow_addons/custom_ops/image:_resampler_ops.so",
        "//tensorflow_addons/custom_ops/image:_distance_ops.so",
        "//tensorflow_addons/custom_ops/image:_threshold_ops.so",
        "//tensorflow_addons/utils",
    ],
)

filegroup(
    name = "sparse_image_warp_test_data",
    srcs = glob(["test_data/*.png"]),
)

py_test(
    name = "distance_ops_test",
    size = "small",
    srcs = [
        "distance_ops_test.py",
    ],
    main = "distance_ops_test.py",
    deps = [
        ":image",
    ],
    srcs_version = "PY2AND3"
)

py_test(
    name = "threshold_ops_test",
    size = "small",
    srcs = [
        "threshold_ops_test.py",
    ],
    main = "threshold_ops_test.py",
    deps = [
        ":image",
    ],
    srcs_version = "PY2AND3"
)

py_test(
    name = "dense_image_warp_test",
    size = "small",
    srcs = [
        "dense_image_warp_test.py",
    ],
    main = "dense_image_warp_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "distance_transform_ops_test",
    size = "small",
    srcs = [
        "distance_transform_test.py",
    ],
    main = "distance_transform_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "distort_image_ops_test",
    size = "small",
    srcs = [
        "distort_image_ops_test.py",
    ],
    main = "distort_image_ops_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "filters_test",
    size = "medium",
    srcs = [
        "filters_test.py",
    ],
    main = "filters_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "transform_ops_test",
    size = "medium",
    srcs = [
        "transform_ops_test.py",
    ],
    main = "transform_ops_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "translate_ops_test",
    size = "medium",
    srcs = [
        "translate_ops_test.py",
    ],
    main = "translate_ops_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = [
        "utils_test.py",
    ],
    main = "utils_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "sparse_image_warp_test",
    size = "medium",
    srcs = [
        "sparse_image_warp_test.py",
    ],
    main = "sparse_image_warp_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "interpolate_spline_test",
    size = "medium",
    srcs = [
        "interpolate_spline_test.py",
    ],
    main = "interpolate_spline_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "connected_components_test",
    size = "medium",
    srcs = [
        "connected_components_test.py",
    ],
    main = "connected_components_test.py",
    deps = [
        ":image",
    ],
)

py_test(
    name = "resampler_ops_test",
    size = "medium",
    srcs = [
        "resampler_ops_test.py",
    ],
    main = "resampler_ops_test.py",
    deps = [
        ":image",
    ],
)
