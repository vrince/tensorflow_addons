package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "libtensorflow_framework",
<<<<<<< HEAD
    srcs = [":libtensorflow_framework.so.2"],
    #data = ["lib/libtensorflow_framework.so.2"],
=======
    srcs = ["%{TF_SHARED_LIBRARY_NAME}"],
>>>>>>> e17ab4a697b97723f2f54715bdb21d2f4d4d313f
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}