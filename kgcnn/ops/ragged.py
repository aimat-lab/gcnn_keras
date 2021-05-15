import tensorflow as tf

from kgcnn.ops.partition import kgcnn_ops_change_partition_type

class DummyRankOneRaggedTensor:
    """Dummy python class. Can not be passed between layers."""

    def __init__(self):
        self.values = None
        self.row_splits = None
        self._row_lengths = None
        self._value_rowids = None

    def from_row_splits(self, values, part):
        self.values = values
        self.row_splits = part
        self._row_lengths = kgcnn_ops_change_partition_type(part, "row_splits", "row_length")
        self._value_rowids = kgcnn_ops_change_partition_type(part, "row_splits", "value_rowids")

    def from_row_lengths(self, values, part):
        self.values = values
        self.row_splits = kgcnn_ops_change_partition_type(part, "row_length", "row_splits")
        self._row_lengths = part
        self._value_rowids = kgcnn_ops_change_partition_type(part, "row_length", "value_rowids")

    def from_value_rowids(self, values, part):
        self.values = values
        self.row_splits = kgcnn_ops_change_partition_type(part, "value_rowids", "row_splits")
        self._row_lengths = kgcnn_ops_change_partition_type(part, "value_rowids", "row_length")
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