import logging


def merge_metrics(m1, m2):
    """Merge two metric lists or dicts of `ks.metrics` objects."""
    if m1 is None:
        return m2
    if m2 is None:
        return m1
    # Dict case with multiple named outputs.
    if isinstance(m1, dict) and isinstance(m2, dict):
        keys = set(list(m1.keys()) + list(m2.keys()))
        m = {key: [] for key in keys}
        for mu in [m1, m2]:
            for key, value in mu.items():
                if value is not None:
                    m[key] = m[key] + (list(value) if isinstance(value, (list, tuple)) else [value])
        return m
    # Lists for single model output.
    m1 = [m1] if not isinstance(m1, (list, tuple)) else m1
    m2 = [m2] if not isinstance(m2, (list, tuple)) else m2
    if all([not isinstance(x1, (list, tuple)) for x1 in m1] + [not isinstance(x2, (list, tuple)) for x2 in m2]):
        return m1 + m2
    # List for multiple model output with nested lists.
    if len(m1) == len(m2):
        m = [[]] * len(m1)
        for i in range(len(m)):
            for mu in [m1, m2]:
                if mu[i] is not None:
                    m[i] = m[i] + (list(mu[i]) if isinstance(mu[i], (list, tuple)) else [mu[i]])
        return m
    else:
        logging.error("For multiple model outputs require same length of metrics list to merge.")
    logging.error("Can not merge metrics '%s' and '%s'." % (m1, m2))
    return None