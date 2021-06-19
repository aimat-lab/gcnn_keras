import tensorflow as tf

from kgcnn.ops.partition import change_partition_by_name

class DummyRankOneRaggedTensor:
    """Dummy python class. Can not be passed between layers. Does not inherit from Tensor or CompositeTensor."""

    def __init__(self):
        self.values = None
        self.row_splits = None
        self._row_lengths = None
        self._value_rowids = None

    def from_row_splits(self, values, part):
        self.values = values
        self.row_splits = part
        self._row_lengths = change_partition_by_name(part, "row_splits", "row_length")
        self._value_rowids = change_partition_by_name(part, "row_splits", "value_rowids")

    def from_row_lengths(self, values, part):
        self.values = values
        self.row_splits = change_partition_by_name(part, "row_length", "row_splits")
        self._row_lengths = part
        self._value_rowids = change_partition_by_name(part, "row_length", "value_rowids")

    def from_value_rowids(self, values, part):
        self.values = values
        self.row_splits = change_partition_by_name(part, "value_rowids", "row_splits")
        self._row_lengths = change_partition_by_name(part, "value_rowids", "row_length")
        self._value_rowids = part

    def row_lengths(self):
        return self._row_lengths

    def value_rowids(self):
        return self._value_rowids

    def from_values_partition(self,values, part, partition_type):
        if partition_type in ["row_length", "row_lengths"]:
            self.from_row_lengths(values, part)
        elif partition_type == "row_splits":
            self.from_row_splits(values, part)
        elif partition_type == "value_rowids":
            self.from_value_rowids(values, part)
        else:
            raise TypeError("Error: Unknown partition scheme, use: 'row_lengths', 'row_splits', 'value_rowids'.")