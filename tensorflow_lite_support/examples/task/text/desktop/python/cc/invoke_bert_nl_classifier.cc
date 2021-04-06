#include <iostream>
#include <limits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/category.h"
#include "tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h"

ABSL_FLAG(std::string, model_path, "",
          "Absolute path to the '.tflite' bert classification model.");
ABSL_FLAG(std::string, text, "", "Text to classify.");

namespace tflite {
namespace task {
namespace text {
namespace nlclassifier {

std::unique_ptr<BertNLClassifier> classifier;

void InitializeModel(int argc, char** argv){
  // Initialization
  classifier = BertNLClassifier::CreateFromFile(argv[0]).value(); //model_path
}


void RunInference(int argc, char** argv, char **strings){
  // Run inference
  std::vector<core::Category> categories =
      classifier->Classify(argv[0]); //input text

  // for (int i = 0; i < categories.size(); ++i) {
  //   const core::Category& category = categories[i];
  //   std::cout << absl::StrFormat("category[%d]: '%s' : '%.5f'\n", i,
  //                                category.class_name, category.score);
  // }
    strcpy(strings[0], ((categories[0]).class_name).c_str());
    strcpy(strings[1], (std::to_string((categories[0]).score).c_str()));
    strcpy(strings[2], ((categories[1]).class_name).c_str());
    strcpy(strings[3], (std::to_string((categories[1]).score).c_str()));
}

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace tflite

extern "C" {
  void InvokeInitializeModel(int argc, char** argv);
  void InvokeRunInference(int argc, char** argv, char **strings);
}

void InvokeInitializeModel(int argc, char** argv){
  tflite::task::text::nlclassifier::InitializeModel(argc, argv);
}

void InvokeRunInference(int argc, char** argv, char **strings){
  tflite::task::text::nlclassifier::RunInference(argc, argv, strings);
}
