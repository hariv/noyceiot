class SparsityParams:
    def __init__(self):
        self.arg_names = ["sparsity_level_setting_mode", "schedule", "sparsity_target", "sparsity_target_epoch",
                          "sparsity_freeze_epoch", "update_per_optimizer_step", "steps_per_epoch", "multistep_steps",
                          "multistep_sparsity_levels", "patience", "power", "concave", "weight_importance"]
        self.present = False
        
