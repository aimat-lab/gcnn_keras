from tensorflow import RaggedTensor

def enabled_ragged(fn, ragged_validate=False):
    def ragged_fn(self, x):
        if isinstance(x, RaggedTensor) and x.ragged_rank == 1:
            out = fn(self, x.values)
            return RaggedTensor.from_row_splits(out, x.row_splits, validate=ragged_validate)
        else:
            return fn(self, x)
    return ragged_fn

