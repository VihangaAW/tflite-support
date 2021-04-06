#include <iostream>
#include <limits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h"


namespace tflite {
namespace task {
namespace text {
namespace qa {

std::unique_ptr<QuestionAnswerer> answerer;

void InitializeModel(int argc, char** argv){
  // Initialization
  answerer = BertQuestionAnswerer::CreateFromFile(argv[0]).value();
}


void RunInference(int argc, char** argv){
  // // Run inference
  std::vector<QaAnswer> answers = answerer->Answer(argv[0], argv[1]); //context_of_question, question_to_ask
  
  for (int i = 0; i < answers.size(); ++i) {
    const QaAnswer& answer = answers[i];
    std::cout << absl::StrFormat(
        "answer[%d]: '%s'\n    logit: '%.5f, start_index: %d, end_index: %d\n",
        i, answer.text, answer.pos.logit, answer.pos.start, answer.pos.end);
  }
}





}  // namespace qa
}  // namespace text
}  // namespace task
}  // namespace tflite

extern "C" {
  void InvokeInitializeModel(int argc, char** argv);
  void InvokeRunInference(int argc, char** argv);
}

void InvokeInitializeModel(int argc, char** argv){
  tflite::task::text::qa::InitializeModel(argc, argv);
}

void InvokeRunInference(int argc, char** argv){
  tflite::task::text::qa::RunInference(argc, argv);
}
