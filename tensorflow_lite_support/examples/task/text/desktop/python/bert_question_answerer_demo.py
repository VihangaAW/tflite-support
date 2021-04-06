import ctypes
import time
import argparse

def Classify(model_path, question, context):
    print("Hello")
    sharedLib = ctypes.CDLL('/home/vihanga/Vihanga/Tflite/tflite-support/bazel-bin/tensorflow_lite_support/examples/task/text/desktop/python/cc/libinvoke_bert_question_answerer.so')
    print("Hello1")

    # Initialize a classifier
    sharedLib.InvokeInitializeModel.restype = None
    sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 1)(bytes(model_path, encoding='utf-8'))
    sharedLib.InvokeInitializeModel(len(args),args)
    print("Hello2")

    sharedLib.InvokeRunInference.restype = None
    sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 2)(bytes(context, encoding='utf-8'),bytes(question, encoding='utf-8'))
    sharedLib.InvokeRunInference(len(args),args)
    print(str(question_to_ask))

def main():
    # Create the parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--context', required=True)
    args = parser.parse_args()
    Classify(args.model_path, args.question, args.context)

if __name__ == '__main__':
    main()
