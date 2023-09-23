from typing import Union


def compare_static_shapes(found: Union[tuple, None], expected: Union[tuple, None]):
    if found is None and expected is None:
        return True
    elif found is None and expected is not None:
        return False
    elif found is not None and expected is None:
        return True
    elif len(found) != len(expected):
        return False
    shapes_okay = []
    for f, e in zip(found, expected):
        if f is None and e is not None:
            shapes_okay.append(False)
        elif f is not None and e is None:
            shapes_okay.append(True)
        elif f is None and e is None:
            shapes_okay.append(True)
        elif f == e:
            shapes_okay.append(True)
        else:
            shapes_okay.append(False)
    return all(shapes_okay)
