from compression import Compression

class Quantization(Compression):
    def __init__(self, algorithm):
        super(Quantization, self).__init__(algorithm)
        self.arg_names.extend(["preset", "quantize_inputs", "quantize_outputs", "export_to_onnx",
                               "overflow_fix", "compression_lr_multiplier"])
    

