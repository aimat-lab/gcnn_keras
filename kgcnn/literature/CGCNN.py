import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.conv.cgcnn import CGCNNLayer
from kgcnn.layers.geom import DisplacementVectorsASU, DisplacementVectorsUnitCell, FracToRealCoords, NodePosition, EuclideanNorm, GaussBasisLayer
from kgcnn.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.mlp import MLP
from kgcnn.utils.models import update_model_kwargs

model_default = {'name': 'CGCNN',
                 'inputs': {'atom_attributes':
                                {'shape': (None,), 'name': 'atom_attributes', 'dtype': 'int64', 'ragged': True},
                            'frac_coords':
                                {'shape': (None, 3), 'name': 'frac_coords', 'dtype': 'float64', 'ragged': True},
                            'multiplicities':
                                {'shape': (None,), 'name': 'multiplicities', 'dtype': 'int64', 'ragged': True},
                            'cell_translations':
                                {'shape': (None, 3), 'name': 'cell_translations', 'dtype': 'float32', 'ragged': True},
                            'symmops':
                                {'shape': (None, 4, 4), 'name': 'symmops', 'dtype': 'float64', 'ragged': True},
                            'lattice_matrix':
                                {'shape': (3, 3), 'name': 'lattice_matrix', 'dtype': 'float64'},
                            'edge_indices':
                               {'shape': (None, 2), 'name': 'edge_indices', 'dtype': 'int64', 'ragged': True},
                            'edge_distances':
                               {'shape': (None,), 'name': 'edge_distances', 'dtype': 'float64', 'ragged': True}},
                 'input_embedding': {'node': {'input_dim': 95, 'output_dim': 64}},
                 'representation': 'unit', # None, 'asu' or 'unit'
                 'make_distances': True,
                 'conv_layer_args': {
                        'units': 64,
                        'activation_s': 'softplus',
                        'activation_out': 'softplus',
                        'batch_normalization': True,
                     },
                 'expand_distance': True,
                 'gauss_args': {'bins': 40, 'distance': 8, 'offset': 0.0, 'sigma': 0.4},
                 'node_pooling_args': {'pooling_method': 'mean'},
                 'depth': 3,
                 'output_mlp': {'use_bias': [True, False], 'units': [64, 1],
                                'activation': ['softplus', 'linear']},
                 }


@update_model_kwargs(model_default)
def make_model(inputs,
        representation,
        make_distances,
        input_embedding,
        conv_layer_args,
        expand_distance,
        depth,
        gauss_args,
        node_pooling_args,
        output_mlp,
        name):
    
    assert 'atom_attributes' in inputs.keys()
    assert 'frac_coords' in inputs.keys()
    assert 'lattice_matrix' in inputs.keys()
    assert 'edge_indices' in inputs.keys()
    
    atom_attributes = ks.layers.Input(**inputs['atom_attributes'])
    edge_indices = ks.layers.Input(**inputs['edge_indices'])
    
    if make_distances:
        
        frac_coords = ks.layers.Input(**inputs['frac_coords'])
        lattice_matrix = ks.layers.Input(**inputs['lattice_matrix'])

        if representation == 'unit':
            assert 'cell_translations' in inputs.keys()
            cell_translations = ks.layers.Input(**inputs['cell_translations'])
            
            displacement_vectors = DisplacementVectorsUnitCell()([frac_coords, edge_indices, cell_translations])
        elif representation == 'asu':
            assert 'cell_translations' in inputs.keys()
            cell_translations = ks.layers.Input(**inputs['cell_translations'])
            assert 'multiplicities' in inputs.keys()
            multiplicities_inp = ks.layers.Input(**inputs['multiplicities'])
            multiplicities = tf.cast(tf.expand_dims(multiplicities_inp, -1), 'float32')
            assert 'symmops' in inputs.keys()
            symmops = ks.layers.Input(**inputs['symmops'])
            
            displacement_vectors = DisplacementVectorsASU()([frac_coords, edge_indices, symmops, cell_translations])
        else:
            x_in, x_out = NodePosition()([frac_coords, edge_indices])
            displacement_vectors = x_out - x_in
            
        displacement_vectors = FracToRealCoords()([displacement_vectors, lattice_matrix])
        
        edge_distances = EuclideanNorm(axis=2, keepdims=True)(displacement_vectors)
        
    else:
        edge_distances_input = ks.layers.Input(**inputs['edge_distances'])
        edge_distances = tf.expand_dims(edge_distances_input, axis=-1)

    if expand_distance:
        edge_distances = GaussBasisLayer(**gauss_args)(edge_distances)

    # embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                                use_embedding=len(inputs['atom_attributes']['shape']) < 2)(atom_attributes)

    for _ in range(depth):
        n = CGCNNLayer(**conv_layer_args)([n, edge_distances, edge_indices])

    if representation == 'asu':
        out = PoolingWeightedNodes(**node_pooling_args)([n, multiplicities])
    else:
        out = PoolingNodes(**node_pooling_args)(n)
    
    out = MLP(**output_mlp)(out)

    if make_distances:
        if representation == 'unit':
            model = ks.models.Model(
                    inputs=[atom_attributes, frac_coords, cell_translations, lattice_matrix, edge_indices],
                    outputs=out, name=name)
        elif representation == 'asu':
            model = ks.models.Model(
                    inputs=[atom_attributes, frac_coords, multiplicities_inp, cell_translations,
                        symmops, lattice_matrix, edge_indices], outputs=out, name=name)
        else:
            model = ks.models.Model(
                    inputs=[atom_attributes, frac_coords, lattice_matrix, edge_indices],
                    outputs=out, name=name)
    else:
        model = ks.models.Model(
                inputs=[atom_attributes, edge_distances_input, edge_indices],
                outputs=out, name=name)
    return model

