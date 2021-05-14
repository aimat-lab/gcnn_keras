import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.models import generate_mol_graph_input

# Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
# Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann
# https://arxiv.org/abs/2011.14115

def make_dimnet_pp(
        # Input
        input_node_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3):

    node_input, n, xyz_input, bond_index_input, angle_index_input, _ = generate_mol_graph_input(input_node_shape,
                                                                                                [None, 3],
                                                                                                [None, 2],
                                                                                                [None, 2])


    raise NotImplementedError("Model not yet tested")

    main_output = n
    model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input],
                                  outputs=main_output)

    return model