package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# To create the shared library
cc_binary(
    name = "invoke_nl_classifier",
    srcs = ["invoke_nl_classifier.cc"],
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core:category",
        "//tensorflow_lite_support/cc/task/text/nlclassifier:nl_classifier",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
    linkshared = 1,     ## important
)

# To create the shared library
cc_binary(
    name = "invoke_bert_question_answerer",
    srcs = ["invoke_bert_question_answerer.cc"],
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/text/qa:bert_question_answerer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
    linkshared = 1,     ## important
)

# To create the shared library
cc_binary(
    name = "invoke_bert_nl_classifier",
    srcs = ["invoke_bert_nl_classifier.cc"],
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core:category",
        "//tensorflow_lite_support/cc/task/text/nlclassifier:bert_nl_classifier",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
    linkshared = 1,     ## important
)
