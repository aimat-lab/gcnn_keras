from kgcnn.data.visual_graph import VisualGraphDataset


class VgdMockDataset(VisualGraphDataset):

    def __init__(self):
        super(VgdMockDataset, self).__init__(name='mock')
