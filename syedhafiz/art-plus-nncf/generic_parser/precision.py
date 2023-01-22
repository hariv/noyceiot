class Precision:
    def __init__(self):
        self.arg_names = ["type", "bits", "num_data_points", "iter_number", "tolerance", "compression_ratio",
                          "eval_subset_ratio", "warmup_iter_number", "bitwidth_per_scope",
                          "traces_per_layer_path", "dump_init_precision_data", "bitwidth_assignment_mode"]
        self.present = False
