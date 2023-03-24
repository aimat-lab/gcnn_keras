import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer


class AtomEmbedding(GraphBaseLayer):
    """Emebedding layer for atomic number and other other element type properties."""

    default_atomic_number_embedding_args = {"input_dim": 119, "output_dim": 64}

    def __init__(
        self,
        atomic_number_embedding_args=default_atomic_number_embedding_args,
        atomic_mass=None,
        atomic_radius=None,
        electronegativity=None,
        ionization_energy=None,
        oxidation_states=None,
        **kwargs
    ):
        """Initialize the AtomEmbedding Layer.

        Args:
            atomic_number_embedding_args (dict, optional): Embedding arguments which get passed
                to the keras `Embedding` layer for the atomic number.
                Defaults to self.default_atomic_number_embedding_args.
            atomic_mass (list, optional): List of atomic mass ordered by the atomic number.
                If it is not None, the atomic mass gets included in the embedding, otherwise not.
                Defaults to None.
            atomic_radius (list, optional): List of atomic radius ordered by the atomic number.
                If it is not None, the atomic radius gets included in the embedding, otherwise not.
                Defaults to None.
            electronegativity (list, optional): List of electronegativities ordered by the atomic number.
                If it is not None, the electronegativities gets included in the embedding, otherwise not.
                Defaults to None.
            ionization_energy (list, optional): List of ionization energies ordered by the atomic number.
                If it is not None, the ionization energies  gets included in the embedding, otherwise not.
                Defaults to None.
            oxidation_states (list, optional): List of oxidation states ordered by the atomic number.
                If it is not None, the oxidation states gets included in the embedding, otherwise not.
                Defaults to None.
        """

        super().__init__(**kwargs)

        self.atomic_mass = (
            tf.constant(atomic_mass, dtype=float) if atomic_mass else None
        )
        self.atomic_radius = (
            tf.constant(atomic_radius, dtype=float) if atomic_radius else None
        )
        self.electronegativity = (
            tf.constant(electronegativity, dtype=float) if electronegativity else None
        )
        self.ionization_energy = (
            tf.constant(ionization_energy, dtype=float) if ionization_energy else None
        )
        self.oxidation_states = (
            tf.constant(oxidation_states, dtype=float) if oxidation_states else None
        )

        self.atomic_number_embedding_args = atomic_number_embedding_args
        self.atomic_number_embedding_layer = ks.layers.Embedding(
            **self.atomic_number_embedding_args
        )

    def call(self, inputs):
        atomic_numbers = inputs
        idxs = atomic_numbers - 1  # Shifted by one (zero-indexed)

        feature_list = []
        atomic_number_embedding = self.atomic_number_embedding_layer(idxs)
        feature_list.append(atomic_number_embedding)
        if self.atomic_mass is not None:
            atomic_mass = tf.expand_dims(tf.gather(self.atomic_mass, idxs), -1)
            feature_list.append(atomic_mass)
        if self.atomic_radius is not None:
            atomic_radius = tf.expand_dims(tf.gather(self.atomic_radius, idxs), -1)
            feature_list.append(atomic_radius)
        if self.electronegativity is not None:
            electronegativity = tf.expand_dims(
                tf.gather(self.electronegativity, idxs), -1
            )
            feature_list.append(electronegativity)
        if self.ionization_energy is not None:
            ionization_energy = tf.expand_dims(
                tf.gather(self.ionization_energy, idxs), -1
            )
            feature_list.append(ionization_energy)
        if self.oxidation_states is not None:
            oxidation_states = tf.gather(self.oxidation_states, idxs)
            feature_list.append(oxidation_states)
        return tf.concat(feature_list, -1)
