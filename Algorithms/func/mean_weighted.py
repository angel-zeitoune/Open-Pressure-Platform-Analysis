import numpy as np

def mean_weighted(CoPs):
    """
    Calcula la media ponderada de los centros de presión por intensidad.

    Parámetros:
        CoPs (list): Lista de tuplas (x, y, intensidad)

    Retorna:
        (x_mean, y_mean): Media ponderada de x e y
    """
    CoPs = np.asarray(CoPs)
    
    if CoPs.size == 0:
        return (np.nan, np.nan)

    # Eliminar filas con cualquier NaN
    valid_mask = ~np.isnan(CoPs).any(axis=1)
    CoPs = CoPs[valid_mask]

    if CoPs.size == 0:
        return (np.nan, np.nan)

    w = CoPs[:, 2]
    total_w = np.sum(w)
    if total_w == 0:
        return (np.nan, np.nan)

    weighted_sum = np.sum(CoPs[:, :2] * w[:, None], axis=0)
    return tuple(weighted_sum / total_w)
