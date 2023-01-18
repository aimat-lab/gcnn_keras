from kgcnn.data.visual_graph import VisualGraphDataset


class VgdMockDataset(VisualGraphDataset):
    r"""Synthetic classification dataset containing 100 small, randomly generated graphs,
    where half of them were seeded with a triangular subgraph motif,
    which is the explanation ground truth for the target class distinction.

    """

    def __init__(self):
        super(VgdMockDataset, self).__init__(name='mock')
