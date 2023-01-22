class InputInfo:
    
    def __init__(self, sample_size_str):
        self.sample_size = list(map(lambda x: int(x), sample_size_str.split(",")))
        self.arg_names = ["sample_size", "type", "filler", "keyword"]
        self.present = True

        
