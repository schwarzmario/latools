
import awkward as ak

class CountTask:
    def __init__(self, fcn, *, name=None):
        self.fcn = fcn
        self.name = name
    def initialize(self):
        self.counter: int = 0
    def __call__(self, x, _):
        mask = self.fcn(x)
        if mask.ndim != 1: # remove in case multi-dim counting also makes sense
            raise RuntimeError(f"Counter requires 1-dim (I think), got {mask.ndim}")
        self.counter += ak.count_nonzero(mask)
        return False # TODO: add also a min_entries threshold, so this task can break earlier if possible
    def finalize(self):
        if self.name:
            print(f"Counter {self.name}: {self.counter}")