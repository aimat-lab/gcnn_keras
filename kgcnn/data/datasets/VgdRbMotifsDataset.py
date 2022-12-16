from kgcnn.data.visual_graph import VisualGraphDataset


class VgdRbMotifsDataset(VisualGraphDataset):
    """
    Synthetic graph regression dataset containing 5000 small, randomly generated graphs,
    where some graphs were seeded with special red- and blue-dominant subgraph motifs. Blue values add
    a negative contribution to the graph's overall regression value, while red motifs contribute positively.
    These sub graph motifs present the ground truth explanations for this task.
    """

    def __init__(self):
        super(VgdRbMotifsDataset, self).__init__(name='rb_dual_motifs')
