class QuantizationParams:
    def __init__(self):
        self.arg_names = ["batch_multiplier", "activations_quant_start_epoch",
                          "weights_quant_start_epoch", "lr_poly_drop_start_epoch",
                          "lr_poly_drop_duration_epochs", "disable_wd_start_epoch",
                          "base_lr", "base_wd"]
        self.present = False
        
