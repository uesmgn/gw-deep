class ToIndex:
    def __init__(self, targets):
        self.to_index = lambda x: targets.index(x) if x in targets else None

    def __call__(self, x):
        idx = self.to_index(x)
        if idx is None:
            return -1
        return idx
