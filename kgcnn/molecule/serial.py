from kgcnn.molecule.encoder import OneHotEncoder


def deserialize_encoder(encoder_identifier):
    """Deserialization of encoder class.

    Args:
        encoder_identifier: Identifier, class or function of an encoder.

    Returns:
        obj: Deserialized encoder.
    """
    # TODO: Can extend deserialization to any callable encoder.
    if isinstance(encoder_identifier, dict):
        if encoder_identifier["class_name"] == "OneHotEncoder":
            return OneHotEncoder.from_config(encoder_identifier["config"])
    elif hasattr(encoder_identifier, "__call__"):
        return encoder_identifier
    else:
        raise ValueError("Unable to deserialize encoder %s " % encoder_identifier)