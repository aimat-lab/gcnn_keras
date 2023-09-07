

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
