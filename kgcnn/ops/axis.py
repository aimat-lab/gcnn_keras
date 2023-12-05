def broadcast_shapes(shape1, shape2):
    """Broadcast input shapes to a unified shape.

    Convert to list for mutability.

    Args:
        shape1: A tuple or list of integers.
        shape2: A tuple or list of integers.

    Returns:
        output_shape (list of integers or `None`): The broadcasted shape.

    Example:
    >>> broadcast_shapes((5, 3), (1, 3))
    [5, 3]
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    origin_shape1 = shape1
    origin_shape2 = shape2

    if len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    output_shape = list(shape1)
    for i in range(len(shape1)):
        if shape1[i] == 1:
            output_shape[i] = shape2[i]
        elif shape1[i] is None:
            output_shape[i] = None if shape2[i] == 1 else shape2[i]
        else:
            if shape2[i] == 1 or shape2[i] is None or shape2[i] == shape1[i]:
                output_shape[i] = shape1[i]
            else:
                raise ValueError(
                    "Cannot broadcast shape, the failure dim has value "
                    f"{shape1[i]}, which cannot be broadcasted to {shape2[i]}. "
                    f"Input shapes are: {origin_shape1} and {origin_shape2}."
                )

    return output_shape


# Found in ops/array_ops and copied here for static reference
def get_positive_axis(axis, ndims, axis_name="axis", ndims_name="ndims"):
    """Validate an `axis` parameter, and normalize it to be positive.
    If `ndims` is known (i.e., not `None`), then check that `axis` is in the
    range `-ndims <= axis < ndims`, and return `axis` (if `axis >= 0`) or
    `axis + ndims` (otherwise).
    If `ndims` is not known, and `axis` is positive, then return it as-is.
    If `ndims` is not known, and `axis` is negative, then report an error.

    Args:
        axis: An integer constant
        ndims: An integer constant, or `None`
        axis_name: The name of `axis` (for error messages).
        ndims_name: The name of `ndims` (for error messages).

    Returns:
        The normalized `axis` value.

    Raises:
        ValueError: If `axis` is out-of-bounds, or if `axis` is negative and `ndims is None`.
    """
    if not isinstance(axis, int):
        raise TypeError("%s must be an int; got %s" %
                        (axis_name, type(axis).__name__))
    if ndims is not None:
        if 0 <= axis < ndims:
            return axis
        elif -ndims <= axis < 0:
            return axis + ndims
        else:
            raise ValueError("%s=%s out of bounds: expected %s<=%s<%s" % (axis_name, axis, -ndims, axis_name, ndims))
    elif axis < 0:
        raise ValueError("%s may only be negative if %s is statically known." % (axis_name, ndims_name))
    return axis
