class Compression:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.arg_names = ["algorithm", "ignored_scopes", "target_scopes"]
        self.present = True
