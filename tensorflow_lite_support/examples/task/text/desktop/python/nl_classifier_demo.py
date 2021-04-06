import ctypes
import time
import argparse

def Classify(model_path, text, input_tensor_name, output_score_tensor_name):
    """

    Classify is used to execute the model with not preprocessed input and get the output.

    Parameters: 
    input                   (string): input text

    Returns
    list 

    """
    sharedLib = ctypes.CDLL('/home/vihanga/Vihanga/Tflite/tflite-support/bazel-bin/tensorflow_lite_support/examples/task/text/desktop/python/cc/libinvoke_nl_classifier.so')
    # Initialize a classifier
    sharedLib.InvokeInitializeModel.restype = None
    sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 4)(bytes(model_path, encoding='utf-8'), bytes(input_tensor_name, encoding='utf-8'), bytes(output_score_tensor_name, encoding='utf-8'))
    sharedLib.InvokeInitializeModel(len(args),args)


    sharedLib.InvokeRunInference.restype = ctypes.c_char_p
    sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)

    string_buffers = [ctypes.create_string_buffer(8) for i in range(4)]
    pointers = (ctypes.c_char_p*4)(*map(ctypes.addressof, string_buffers))

    args = (ctypes.c_char_p * 1)(bytes(text, encoding='utf-8'))

    # Run the inference
    sharedLib.InvokeRunInference(len(args), args, pointers)
    results = [(s.value).decode('utf-8') for s in string_buffers]
    
    print(results)


def main():
    # Create the parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--input_tensor_name', required=True)
    parser.add_argument('--output_score_tensor_name', required=True)
    args = parser.parse_args()
    Classify(args.model_path, args.text, args.input_tensor_name, args.output_score_tensor_name)

if __name__ == '__main__':
    main()