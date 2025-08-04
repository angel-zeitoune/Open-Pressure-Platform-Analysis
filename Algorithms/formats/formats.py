import numpy as np

def convert_nan_to_none(value):
    """
    Convierte NaN a None para compatibilidad con JSON.
    """
    if isinstance(value, (list, tuple)):
        return [convert_nan_to_none(v) for v in value]
    return None if np.isnan(value) else value

def to_Point2DIntensity(CoPs):
    values = []
    for x_cp, y_cp, total in CoPs:
        values.append({"C": convert_nan_to_none(x_cp), "R": convert_nan_to_none(y_cp), "I": int(total) })
    return values

