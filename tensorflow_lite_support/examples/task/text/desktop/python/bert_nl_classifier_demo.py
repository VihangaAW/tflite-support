import ctypes
import time

class BertNLClassifier:
  def __init__(self, model_path):
    self.model_path = model_path
    self.sharedLib = ctypes.CDLL('./assets/libinvoke_bert_nl_classifier.so')

    # Initialize a classifier
    self.sharedLib.InvokeInitializeModel.restype = None
    self.sharedLib.InvokeInitializeModel.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    args = (ctypes.c_char_p * 1)(bytes(model_path, encoding='utf-8'))
    self.sharedLib.InvokeInitializeModel(len(args),args)



  @classmethod
  def CreateFromFile(cls, model_path):
      """

      CreateFromFile is used to invoke the interpreter.

      Parameters:
      model_path               (string): tflite model path 
      
      Returns: 
      BertNLClassifier: object 

      """
      return cls(model_path)

  def Classify(self,input):
      """

      Classify is used to execute the model with not preprocessed input and get the output.

      Parameters: 
      input                   (string): input text

      Returns
      list 

      """
      self.sharedLib.InvokeRunInference.restype = ctypes.c_char_p
      self.sharedLib.InvokeRunInference.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)

      string_buffers = [ctypes.create_string_buffer(8) for i in range(4)]
      pointers = (ctypes.c_char_p*4)(*map(ctypes.addressof, string_buffers))
      # Run the inference
      args = (ctypes.c_char_p * 1)(bytes(input, encoding='utf-8'))
      self.sharedLib.InvokeRunInference(len(args), args, pointers)
      results = [(s.value).decode('utf-8') for s in string_buffers]
      return results  

# Testing
timeStart = time.time()
bertNLClassifier = BertNLClassifier.CreateFromFile("/tmp/mobilebert_quantized.tflite")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print("Time spent for loading the model    : " + str(executedTime*1000)+" milliseconds")

timeStart = time.time()
output = bertNLClassifier.Classify("It was a great movie")
timeEnd = time.time()
executedTime = timeEnd-timeStart
print("Time spent for running the inference: " + str(executedTime*1000)+" milliseconds")
print("Output: " + str(output))