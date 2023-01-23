from compression import Compression

class Sparsity(Compression):
    def __init__(self, algorithm):
        super(Sparsity, self).__init__(algorithm)
        self.arg_names = ["algorithm", "sparsity_init", "initializer", "params", "ignored_scopes", "target_scopes",
                          "compression_lr_multiplier"]
