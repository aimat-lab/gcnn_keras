import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.mlp import GraphMLP
from kgcnn.layers.conv.dimenet_conv import ResidualLayer
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='MXMGlobalMP')
class MXMGlobalMP(GraphBaseLayer):

    def __init__(self, units: int =63, **kwargs):
        """Initialize layer."""
        super(MXMGlobalMP, self).__init__(**kwargs)
        self.dim = units
        self.h_mlp = GraphMLP(self.dim)

        self.res1 = ResidualLayer(self.dim)
        self.res2 = ResidualLayer(self.dim)
        self.res3 = ResidualLayer(self.dim)
        self.mlp = GraphMLP([self.dim])

        self.x_edge_mlp = GraphMLP([self.dim * 3, self.dim])
        self.linear = nn.Linear(self.dim, self.dim, bias=False)

    def build(self, input_shape):
        """Build layer."""
        super(MXMGlobalMP, self).build(input_shape)

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        num_edge = edge_attr.size()[0]

        x_edge = torch.cat((x_i[:num_edge], x_j[:num_edge], edge_attr), -1)
        x_edge = self.x_edge_mlp(x_edge)

        x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_j

    def call(self, inputs, **kwargs):
        h, edge_attr, edge_index = inputs

        res_h = h

        # Integrate the Cross Layer Mapping inside the Global Message Passing
        h = self.h_mlp(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)

        # Update function f_u
        h = self.res1(h)
        h = self.mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Message Passing operation
        h = self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr)
        return edges_corrected