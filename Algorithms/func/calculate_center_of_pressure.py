import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from func.mean_weighted import mean_weighted


def calculate_center_of_pressure(matrix):
    """
    Calcula el centro de presión (centroide) de una matriz 2D de presiones.

    Parámetros:
        matriz (np.ndarray): Matriz 2D de presiones (floats o ints)

    Retorna:
        (x, y, total): Coordenadas del centro de presión (col, fila) y el total de presión
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Debe proporcionar una matriz 2D de numpy.")

    total = np.sum(matrix)
    if total == 0:
        return (np.nan, np.nan, 0)  # Evitar división por cero

    h, w = matrix.shape

    # Crear matrices de índices
    y_indices, x_indices = np.indices((h, w))

    x_cp = np.sum(x_indices * matrix) / total
    y_cp = np.sum(y_indices * matrix) / total

    return (x_cp, y_cp, total)


def trajectory_length(CoPs):
    """
    Calcula la longitud de la trayectoria del centro de presión.

    Parámetros:
        CoPs (list): Lista de coordenadas del centro de presión (x, y, total).

    Retorna:
        float: Longitud de la trayectoria.
    """
    CoPs = np.asarray(CoPs)
    mask = ~np.isnan(CoPs).any(axis=1)
    CoP_valid = CoPs[mask]
    xy = CoP_valid[:, :2]
    diffs = np.diff(xy, axis=0)
    distancias = np.linalg.norm(diffs, axis=1)
    traj_length = np.sum(distancias)

    return traj_length, traj_length / len(CoP_valid) if len(CoP_valid) > 0 else 0

def calculate_center_of_pressure_video(data):
    """
    Calcula el centro de presión para un video dado.

    Parámetros:
        data (list): Lista de matrices 2D de presiones.

    Retorna:
        list: Lista de coordenadas del centro de presión para cada frame.
    """
    CoPs = []
    for frame in data:
        x_cp, y_cp, total = calculate_center_of_pressure(frame)
        CoPs.append((x_cp, y_cp, total))

    global_mean = mean_weighted(CoPs)
    traj_length, traj_mean = trajectory_length(CoPs)

    return CoPs, global_mean, traj_length, traj_mean


if __name__ == "__main__":
    # Test 1: matriz simple
    matriz1 = np.array([[1, 2], [3, 10]])
    x_cp, y_cp, total = calculate_center_of_pressure(matriz1)
    print("Test 1:")
    print("Matriz:\n", matriz1)
    print(f"Centro de presión: x={x_cp}, y={y_cp}, total={total}")
    print("Expected: x=0.75, y=0.0.8125, total=10", 'OK' if x_cp == 0.75 and y_cp == 0.8125 and total == 16 else 'FAIL')

    # Test 2: matriz con ceros
    matriz2 = np.zeros((2, 2))
    x_cp, y_cp, total = calculate_center_of_pressure(matriz2)
    print("\nTest 2:")
    print("Matriz:\n", matriz2)
    print(f"Centro de presión: x={x_cp}, y={y_cp}, total={total}")
    print("Expected: x=None, y=None, total=0", 'OK' if np.isnan(x_cp) and np.isnan(y_cp)  and total == 0 else 'FAIL')

    # Test 3: matriz más grande
    matriz3 = np.array([[8, 7, 6], [5, 4, 5], [6, 7, 8]])
    x_cp, y_cp, total = calculate_center_of_pressure(matriz3)
    print("\nTest 3:")
    print("Matriz:\n", matriz3)
    print(f"Centro de presión: x={x_cp}, y={y_cp}, total={total}")
    print("Expected: x=1.0, y=1.0, total=56", 'OK' if x_cp == 1.0 and y_cp == 1.0 and total == 56 else 'FAIL')

    # Test 4: video (lista de matrices)
    print("\nTest 4: calculate_center_of_pressure_video")
    video_data = [
        np.array([[1, 2], [3, 4]]),
        np.array([[6, 4], [2, 7]]),
        np.array([[6, 5], [3, 5]]),
        np.array([[0, 0], [0, 0]])
    ]
    CoPs, global_mean, traj_length = calculate_center_of_pressure_video(video_data)
    print("CoPs:", CoPs)
    print("Global mean:", global_mean)
    print("Trajectory length:", traj_length)
    # Verificación simple: global_mean debe ser una tupla/lista de dos elementos
    print("Expected: len(CoPs)=3, len(global_mean)=2", 'OK' if len(CoPs) == 3 and len(global_mean) == 2 else 'FAIL')
