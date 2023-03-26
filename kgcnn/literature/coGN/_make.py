import tensorflow as tf
from tensorflow.core.protobuf.control_flow_pb2 import ValuesDef
from tensorflow.python.lib.io.file_io import atomic_write_string_to_file
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.coGN._preprocessing_layers import LineGraphAngleDecoder
from kgcnn.literature.coGN._gates import HadamardProductGate
from kgcnn.literature.coGN._graph_network.graph_networks import NestedGraphNetwork, SequentialGraphNetwork, GraphNetwork, \
    GraphNetworkMultiplicityReadout, CrystalInputBlock
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.coGN._embedding_layers._edge_embedding import EdgeEmbedding, SinCosExpansion, GaussBasisExpansion
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP
from tensorflow.keras.layers import GRUCell, LSTMCell
from kgcnn.model.utils import update_model_kwargs
from ._coGN_config import model_default
from ._preprocessing_layers import EdgeDisplacementVectorDecoder

ks = tf.keras


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_block_cfg=None,
               processing_blocks_cfg=None,
               output_block_cfg=None,):
    r"""Make connectivity optimized graph networks for crystals.

    Args:
        inputs (list): List of inputs kwargs.
        input_block_cfg (dict): Input block config.
        processing_blocks_cfg (list): List of processing block configs.
        output_block_cfg: Output block config.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    edge_indices = ks.Input(**inputs['edge_indices'])
    atomic_number = ks.Input(**inputs['atomic_number'])
    if 'cell_translation' in inputs and 'frac_coords' in inputs and 'lattice_matrix' in inputs:
        calculate_edge_offset = True
        preprocessing_layer = EdgeDisplacementVectorDecoder()
        cell_translation = ks.Input(**inputs['cell_translation'])
        frac_coords = ks.Input(**inputs['frac_coords'])
        lattice_matrix = ks.Input(**inputs['lattice_matrix'])
        offset, _, _, _ = preprocessing_layer([cell_translation, frac_coords, lattice_matrix, edge_indices])
    elif 'offset' in inputs:
        calculate_edge_offset = False
        offset = ks.Input(**inputs['offset'])
    else:
        raise ValueError('The model needs either the "offset"\
                         or "cell_translation", "frac_coords" and "lattice_matrix" as input.')

    if 'voronoi_ridge_area' in inputs:
        voronoi_ridge_area = True
        inp_voronoi_ridge_area = ks.Input(**inputs['voronoi_ridge_area'])
    else:
        voronoi_ridge_area = False

    if 'multiplicity' in inputs:
        multiplicity = True
        inp_multiplicity = ks.Input(**inputs['multiplicity'])
        inp_multiplicity_ = tf.cast(inp_multiplicity, tf.float32)
    else:
        multiplicity = False

    if 'line_graph_edge_indices' in inputs:
        line_graph = True
        line_graph_edge_indices = ks.Input(**inputs['line_graph_edge_indices'])
        line_graph_angle_decoder = LineGraphAngleDecoder()
        angle_embedding_layer = GaussBasisExpansion.from_bounds(16, 0, 3.2)
        angles, _, _, _ = line_graph_angle_decoder([None, offset, None, line_graph_edge_indices])
        angle_embeddings = angle_embedding_layer(tf.expand_dims(angles, -1))
    else:
        line_graph = False


    euclidean_norm = EuclideanNorm()
    distance = euclidean_norm(offset)
    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    sequential_gn = SequentialGraphNetwork(
        [GraphNetworkConfigurator.get_gn_block(**cfg) for cfg in processing_blocks_cfg]
    )
    output_block = GraphNetworkConfigurator.get_gn_block(**output_block_cfg)

    if multiplicity:
        node_input = {'features': atomic_number, 'multiplicity': inp_multiplicity_}
    else:
        node_input = atomic_number

    if voronoi_ridge_area:
        edge_input = (distance, inp_voronoi_ridge_area)
    else:
        edge_input = distance

    if line_graph:
        global_input = {'line_graph_edge_indices': line_graph_edge_indices, 'line_graph_edges': angle_embeddings}
    else:
        global_input = None

    edge_features, node_features, _, _ = crystal_input_block([edge_input,
                                                              node_input,
                                                              None,
                                                              edge_indices])
    x = sequential_gn([edge_features, node_features, global_input, edge_indices])
    _, _, out, _ = output_block(x)
    out = output_block.get_features(out)

    edge_inputs, node_inputs, global_inputs = [], [atomic_number], []
    if calculate_edge_offset:
        edge_inputs.append(cell_translation)
        node_inputs.append(frac_coords)
        global_inputs.append(lattice_matrix)
    else:
        edge_inputs.append(offset)
    if multiplicity:
        node_inputs.append(inp_multiplicity)
    if voronoi_ridge_area:
        edge_inputs.append(inp_voronoi_ridge_area)
    if line_graph:
        global_inputs.append(line_graph_edge_indices)

    input_list = edge_inputs + node_inputs, global_inputs + [edge_indices]

    return ks.Model(inputs=input_list, outputs=out)


class GraphNetworkConfigurator():

    def __init__(self, units=64, activation='swish', last_layer_activation='tanh',
                 edge_mlp_depth=3, node_mlp_depth=3, global_mlp_depth=3,
                 nested_edge_mlp_depth=3, nested_node_mlp_depth=3,
                 depth=4, nested_depth=0):
        self.units = units
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.edge_mlp_depth = edge_mlp_depth
        self.node_mlp_depth = node_mlp_depth
        self.global_mlp_depth = global_mlp_depth
        self.nested_edge_mlp_depth = nested_edge_mlp_depth
        self.nested_node_mlp_depth = nested_node_mlp_depth
        self.depth = depth
        self.nested_depth = nested_depth

        self.default_input_block_cfg = {
            'node_size': self.units,
            'edge_size': self.units,
            'atomic_mass': True,
            'atomic_radius': True,
            'electronegativity': True,
            'ionization_energy': True,
            'oxidation_states': True,
            'edge_embedding_args': {
                'bins_distance': 32,
                'max_distance': 5.,
                'distance_log_base': 1.,
                'bins_voronoi_area': None,
                'max_voronoi_area': None}}

        self.default_nested_block_cfg = {
            'edge_mlp': {
                'units': [self.units] * self.nested_edge_mlp_depth,
                'activation': self.get_activations(self.nested_edge_mlp_depth)},
            'node_mlp': {
                'units': [self.units] * self.nested_node_mlp_depth,
                'activation': self.get_activations(self.nested_node_mlp_depth)},
            'global_mlp': None,
            'nested_blocks_cfgs': None,
            'aggregate_edges_local': 'mean',
            'aggregate_edges_global': None,
            'aggregate_nodes': None,
            'return_updated_edges': False,
            'return_updated_nodes': True,
            'return_updated_globals': True,
            'edge_attention_mlp_local': self.attention_cfg,
            'edge_attention_mlp_global': self.attention_cfg,
            'node_attention_mlp': self.attention_cfg,
            'edge_gate': None,
            'node_gate': None,
            'global_gate': None,
            'residual_node_update': False,
            'residual_edge_update': False,
            'residual_global_update': False,
            'update_edges_input': [True, True, True, False],
            'update_nodes_input': [True, False, False],
            'update_global_input': [False, True, False],
            'multiplicity_readout': False}

        nested_blocks_cfgs = None
        if self.nested_depth > 0:
            nested_blocks_cfgs = [self.default_nested_block_cfg for _ in range(self.nested_depth)]
        self.default_processing_block_cfg = {
            'edge_mlp': {
                'units': [self.units] * self.edge_mlp_depth,
                'activation': self.get_activations(self.edge_mlp_depth)},
            'node_mlp': {
                'units': [self.units] * self.node_mlp_depth,
                'activation': self.get_activations(self.node_mlp_depth)},
            'global_mlp': None,
            'nested_blocks_cfgs': nested_blocks_cfgs,
            'aggregate_edges_local': 'sum',
            'aggregate_edges_global': None,
            'aggregate_nodes': None,
            'return_updated_edges': False,
            'return_updated_nodes': True,
            'return_updated_globals': True,
            'edge_attention_mlp_local': self.attention_cfg,
            'edge_attention_mlp_global': self.attention_cfg,
            'node_attention_mlp': self.attention_cfg,
            'edge_gate': None,
            'node_gate': None,
            'global_gate': None,
            'residual_node_update': False,
            'residual_edge_update': False,
            'residual_global_update': False,
            'update_edges_input': [True, True, True, False],
            'update_nodes_input': [True, False, False],
            'update_global_input': [False, True, False],
            'multiplicity_readout': False}

        self.default_output_block_cfg = {
            'edge_mlp': None,
            'node_mlp': None,
            'global_mlp': {
                'units': [self.units] * (self.global_mlp_depth - 1) + [1],
                'activation': self.get_activations(self.global_mlp_depth, last_layer_activation='linear')},
            'nested_blocks_cfgs': None,
            'aggregate_edges_local': 'sum',
            'aggregate_edges_global': None,
            'aggregate_nodes': 'sum',
            'return_updated_edges': False,
            'return_updated_nodes': True,
            'return_updated_globals': True,
            'edge_attention_mlp_local': self.attention_cfg,
            'edge_attention_mlp_global': self.attention_cfg,
            'node_attention_mlp': self.attention_cfg,
            'edge_gate': None,
            'node_gate': None,
            'global_gate': None,
            'residual_node_update': False,
            'residual_edge_update': False,
            'residual_global_update': False,
            'update_edges_input': [True, True, True, False],
            'update_nodes_input': [True, False, False],
            'update_global_input': [False, True, False],
            'multiplicity_readout': False}

    @property
    def attention_cfg(self):
        return {'units': [32, 1], 'activation': [self.activation, self.last_layer_activation]}

    @property
    def input_block_cfg(self):
        return self.default_input_block_cfg

    @property
    def processing_block_cfg(self):
        if self.nested_depth > 0:
            nested_blocks_cfgs = [self.default_nested_block_cfg for _ in range(self.nested_depth)]
            self.default_processing_block_cfg['nested_blocks_cfgs'] = nested_blocks_cfgs
        else:
            self.default_processing_block_cfg['nested_blocks_cfgs'] = None
        return self.default_processing_block_cfg

    @property
    def nested_block_cfg(self):
        return self.default_nested_block_cfg

    @property
    def output_block_cfg(self):
        return self.default_output_block_cfg

    def get_activations(self, depth: int, activation=None, last_layer_activation=None):
        if activation is None:
            activation = self.activation
        if last_layer_activation is None:
            last_layer_activation = self.last_layer_activation
        return [activation] * (depth - 1) + [last_layer_activation]

    @staticmethod
    def get_gn_block(edge_mlp={'units': [64, 64], 'activation': 'swish'},
                     node_mlp={'units': [64, 64], 'activation': 'swish'},
                     global_mlp={'units': [64, 32, 1], 'activation': ['swish', 'swish', 'linear']},
                     nested_blocks_cfgs=None,
                     aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
                     return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
                     edge_attention_mlp_local={'units': [1], 'activation': 'linear'},
                     edge_attention_mlp_global={'units': [1], 'activation': 'linear'},
                     node_attention_mlp={'units': [1], 'activation': 'linear'},
                     edge_gate=None, node_gate=None, global_gate=None,
                     residual_node_update=False, residual_edge_update=False, residual_global_update=False,
                     update_edges_input=[True, True, True, False],  # [edges, nodes_in, nodes_out, globals_]
                     update_nodes_input=[True, False, False],  # [aggregated_edges, nodes, globals_]
                     update_global_input=[False, True, False],  # [aggregated_edges, aggregated_nodes, globals_]
                     multiplicity_readout=False):
        if edge_gate == 'gru':
            edge_gate = GRUCell(edge_mlp['units'][-1])
        elif edge_gate == 'hadamard':
            edge_gate = HadamardProductGate(units=edge_mlp['units'][-1], return_twice=True)
        elif edge_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            edge_gate = None
        if node_gate == 'gru':
            node_gate = GRUCell(node_mlp['units'][-1])
        elif node_gate == 'hadamard':
            node_gate = HadamardProductGate(units=node_mlp['units'][-1], return_twice=True)
        elif node_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            node_gate = None
        if global_gate == 'gru':
            global_gate = GRUCell(global_mlp['units'][-1])
        elif global_gate == 'hadamard':
            global_gate = HadamardProductGate(units=global_mlp['units'][-1], return_twice=True)
        elif global_gate == 'lstm':
            assert False, "LSTM isnt supported yet."
        else:
            global_gate = None

        edge_mlp = MLP(**edge_mlp) if edge_mlp is not None else None
        node_mlp = MLP(**node_mlp) if node_mlp is not None else None
        global_mlp = MLP(**global_mlp) if global_mlp is not None else None
        edge_attention_mlp_local = MLP(**edge_attention_mlp_local) if edge_attention_mlp_local is not None else None
        edge_attention_mlp_global = MLP(**edge_attention_mlp_global) if edge_attention_mlp_global is not None else None
        node_attention_mlp = MLP(**node_attention_mlp) if node_attention_mlp is not None else None

        if nested_blocks_cfgs is not None and multiplicity_readout:
            raise ValueError("Nested GN blocks and multiplicity readout do not work together.")
        if multiplicity_readout:
            block = GraphNetworkMultiplicityReadout(edge_mlp, node_mlp, global_mlp,
                                                    aggregate_edges_local=aggregate_edges_local,
                                                    aggregate_edges_global=aggregate_edges_global,
                                                    aggregate_nodes=aggregate_nodes,
                                                    return_updated_edges=return_updated_edges,
                                                    return_updated_nodes=return_updated_nodes,
                                                    return_updated_globals=return_updated_globals,
                                                    edge_attention_mlp_local=edge_attention_mlp_local,
                                                    edge_attention_mlp_global=edge_attention_mlp_global,
                                                    node_attention_mlp=node_attention_mlp,
                                                    edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                                                    residual_edge_update=residual_edge_update,
                                                    residual_node_update=residual_node_update,
                                                    residual_global_update=residual_global_update,
                                                    update_edges_input=update_edges_input,
                                                    update_nodes_input=update_nodes_input,
                                                    update_global_input=update_global_input)
        elif nested_blocks_cfgs is not None:
            nested_blocks = SequentialGraphNetwork(
                [GraphNetworkConfigurator.get_gn_block(**cfg) for cfg in nested_blocks_cfgs])
            block = NestedGraphNetwork(edge_mlp, node_mlp, global_mlp, nested_blocks,
                                       aggregate_edges_local=aggregate_edges_local,
                                       aggregate_edges_global=aggregate_edges_global,
                                       aggregate_nodes=aggregate_nodes,
                                       return_updated_edges=return_updated_edges,
                                       return_updated_nodes=return_updated_nodes,
                                       return_updated_globals=return_updated_globals,
                                       edge_attention_mlp_local=edge_attention_mlp_local,
                                       edge_attention_mlp_global=edge_attention_mlp_global,
                                       node_attention_mlp=node_attention_mlp,
                                       edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                                       residual_edge_update=residual_edge_update,
                                       residual_node_update=residual_node_update,
                                       residual_global_update=residual_global_update,
                                       update_edges_input=update_edges_input,
                                       update_nodes_input=update_nodes_input,
                                       update_global_input=update_global_input)
        else:
            block = GraphNetwork(edge_mlp, node_mlp, global_mlp,
                                 aggregate_edges_local=aggregate_edges_local,
                                 aggregate_edges_global=aggregate_edges_global,
                                 aggregate_nodes=aggregate_nodes,
                                 return_updated_edges=return_updated_edges,
                                 return_updated_nodes=return_updated_nodes,
                                 return_updated_globals=return_updated_globals,
                                 edge_attention_mlp_local=edge_attention_mlp_local,
                                 edge_attention_mlp_global=edge_attention_mlp_global,
                                 node_attention_mlp=node_attention_mlp,
                                 edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
                                 residual_edge_update=residual_edge_update,
                                 residual_node_update=residual_node_update,
                                 residual_global_update=residual_global_update,
                                 update_edges_input=update_edges_input,
                                 update_nodes_input=update_nodes_input,
                                 update_global_input=update_global_input)
        return block

    @staticmethod
    def get_input_block(node_size=64, edge_size=64,
                        atomic_mass=True, atomic_radius=True, electronegativity=True, ionization_energy=True,
                        oxidation_states=True,
                        edge_embedding_args={
                            'bins_distance': 32, 'max_distance': 5., 'distance_log_base': 1.,
                            'bins_voronoi_area': None, 'max_voronoi_area': None}):
        periodic_table = PeriodicTable()

        atom_embedding_layer = AtomEmbedding(
            atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_size},
            atomic_mass=periodic_table.get_atomic_mass() if atomic_mass else None,
            atomic_radius=periodic_table.get_atomic_radius() if atomic_radius else None,
            electronegativity=periodic_table.get_electronegativity() if electronegativity else None,
            ionization_energy=periodic_table.get_ionization_energy() if ionization_energy else None,
            oxidation_states=periodic_table.get_oxidation_states() if oxidation_states else None)
        edge_embedding_layer = EdgeEmbedding(**edge_embedding_args)
        crystal_input_block = CrystalInputBlock(atom_embedding_layer,
                                                edge_embedding_layer,
                                                atom_mlp=MLP([node_size]), edge_mlp=MLP([edge_size]))
        return crystal_input_block
