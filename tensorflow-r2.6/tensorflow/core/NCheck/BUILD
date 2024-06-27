load("//tensorflow:tensorflow.bzl", "tf_cuda_library")


tf_cuda_library(
    name = "ncheck_persist",
    srcs = ["persist.cc"],
    hdrs = ["persist.h"],
    deps = ["//tensorflow/core/platform:logging",
    ],
    visibility = ["//visibility:public"],
    linkopts = ["-lpmem"],
)





tf_cuda_library(
    name = "tensor_metadata",
    srcs = ["tensor_metadata.cc"],
    hdrs = ["tensor_metadata.h"],
    deps = [
        "//tensorflow/core/NCheck:ncheck_persist",
    ],
    visibility = ["//visibility:public"],
)