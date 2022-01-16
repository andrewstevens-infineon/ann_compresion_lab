
"""Utilities to support model execution/debug using TF-Lite interpreter."""

import tensorflow as tf
import numpy as np



# TODO generalize to N inputs and N outputs by probing sized of _details attributes,

class TfliteModel(object):

    """
    Wrapper for tflite.Interpreter run simple single-input single-output
    tflite model that exposes compatible interface to keras.Model

    Automatically quantizes/dequantizer inputs and outputs as required.\
    By default uses the slooow reference kernels to ensure  a close match
    with tflite(u).
    """
    def __init__(self, model_path=None, model_content=None, tflite_op_type="ref"):

        # TODO should probaby also auto-probe dtype/bits from inputs too!!
        if tflite_op_type == "ref":
            op_type = tf.lite.experimental.OpResolverType.BUILTIN_REF
        elif tflite_op_type == "builtin":
            op_type = tf.lite.experimental.OpResolverType.BUILTIN
        else:
            tflite_op_type = tf.lite.experimental.OpResolverType.AUTO

        self.interp = tf.lite.Interpreter(model_path=model_path, model_content=model_content, experimental_op_resolver_type=op_type)
        # Allocate memory for  model
        self.interp.allocate_tensors()

        in_details = self.interp.get_input_details()
        self.in_type = in_details[0]["dtype"]
        self.in_idx = in_details[0]["index"]

        out_details = self.interp.get_output_details()
        self.out_type =  out_details[0]["dtype"]
        self.out_idx = out_details[0]["index"]


    def predict(self, x):
        
        """
        Run inference on the wrapped quantized tflite model.   if input is missing batch dimension
        this is added.   Takes care to avoid dangling references to
        internal tflite buffers between inferences.
        """
        in_tensor = self.interp.tensor(self.in_idx)()

        if x.shape != in_tensor.shape:
            raise Exception( f"Model input tensor shape {in_tensor.shape} != input shape {x_quant.shape}")
        for i in range(x.shape[0]):
            in_tensor[i] = x[i]
        in_tensor = None
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_idx)
        return y


        
 
