""" Análisis postural estático. """

import sys
import json
import numpy as np

from core.decode_matrix import decode_matrix
from core.read_json_file import read_json_file
from formats.formats import convert_nan_to_none, to_Point2DIntensity
from func.calculate_center_of_pressure import calculate_center_of_pressure_video


def main(input_data = None):
    if input_data is None:
        input_data = sys.stdin.read()
    test = json.loads(input_data)

    pressure_threshold = 1
    rows = test["Platform"]["Rows"]
    cols = test["Platform"]["Columns"]
    frames = test["Platform"]["Frames"]
    total_duration = test.get("Duration", 1) / 1000 # en segundos


    framesData = [decode_matrix(frame["Data"], rows, cols) for frame in frames]
    CoPs, global_mean, traj_length, traj_mean = calculate_center_of_pressure_video(framesData)
    mean_velocity = traj_length / total_duration

    framesDataLeft = [frame[:, :cols // 2] for frame in framesData]
    framesDataRight = [frame[:, cols // 2:] for frame in framesData]
    # Calcular estadísticas de presión
    CoPs_left, global_mean_left, traj_length_left, traj_mean_left = calculate_center_of_pressure_video(framesDataLeft)
    CoPs_right, global_mean_right, traj_length_right, traj_mean_right = calculate_center_of_pressure_video(framesDataRight)
    # Sumar la mitad de las columnas para el centro de presión derecho, solo a nivel de x
    CoPs_right = [(x + cols // 2, y, total) for x, y, total in CoPs_right]
    global_mean_right = (global_mean_right[0] + cols // 2, global_mean_right[1])

    # Inicializar listas
    CoP_global = []
    pressure_total, pressure_left, pressure_right = [], [], []
    contact_area_left, contact_area_right = [], []
    max_pressure_left, max_pressure_right = [], []
    mean_pressure_left, mean_pressure_right = [], []

    # Graficar los framesData[10][:, :cols // 2]
    # import matplotlib.pyplot as plt
    # plt.imshow(framesData[10][:, cols // 2:], cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.title('Left Foot Pressure Distribution')
    # plt.show()


    for data in framesData:
        total_p = np.sum(data)
        pressure_total.append(total_p)

        # Pie izquierdo
        left_mat = data[:, :cols // 2]
        p_left = np.sum(left_mat)
        pressure_left.append(p_left)
        contact_area_left.append(np.sum(left_mat > pressure_threshold) * 0.25)  # np.count_nonzero(frame) * 0.25 cm2
        max_pressure_left.append(np.max(left_mat))
        mean_pressure_left.append(np.mean(left_mat[left_mat > pressure_threshold]) if np.any(left_mat > pressure_threshold) else 0)

        # Pie derecho
        right_mat = data[:, cols // 2:]
        p_right = np.sum(right_mat)
        pressure_right.append(p_right)
        contact_area_right.append(np.sum(right_mat > pressure_threshold) * 0.25) # np.count_nonzero(frame) * 0.25
        max_pressure_right.append(np.max(right_mat))
        mean_pressure_right.append(np.mean(right_mat[right_mat > pressure_threshold]) if np.any(right_mat > pressure_threshold) else 0)

    # Convertir a arrays
    pressure_left = np.array(pressure_left)
    pressure_right = np.array(pressure_right)

    # CoP válidos
    CoP_global = np.array([[float(x), float(y)] for x, y, total in CoPs])
    mask = ~np.isnan(CoP_global[:, 0])
    CoP_valid = CoP_global[mask]

    # Elipse de confianza (95%)
    cov = np.cov(CoP_valid.T)
    vals, vecs = np.linalg.eig(cov)
    major, minor = 2 * np.sqrt(5.991 * vals)
    ellipse_area = np.pi * major * minor

    # Índice de simetría
    simmetry_index = np.mean(np.abs(pressure_left.astype(np.int64) - pressure_right.astype(np.int64)) / (pressure_left + pressure_right + 1e-6))

    # Resultados
    results = {
        "Trayectoria total": convert_nan_to_none(traj_length),
        "Trayectoria media": convert_nan_to_none(traj_mean),
        "Velocidad media": convert_nan_to_none(mean_velocity),

        "Area elipse 95": ellipse_area,
        "Std CoP Medio-Lateral": np.nanstd(CoP_global[:, 0]),
        "Std CoP Antero-Posterior": np.nanstd(CoP_global[:, 1]),

        # Distribución de peso
        "% Peso izq": np.mean(pressure_left / (pressure_left + pressure_right + 1e-8)) * 100 if np.sum(pressure_left + pressure_right) > 0 else 0,
        "% Peso der": np.mean(pressure_right / (pressure_left + pressure_right + 1e-8)) * 100 if np.sum(pressure_left + pressure_right) > 0 else 0,
        "Indice de simetría": simmetry_index,

        # # Presiones
        "Presión máx izq": int(np.max(max_pressure_left)),
        "Presión máx der": int(np.max(max_pressure_right)),
        "Presión med izq": int(np.mean(mean_pressure_left)),
        "Presión med der": int(np.mean(mean_pressure_right)),

        # Áreas de contacto
        "Area de contacto izq (cm2)": np.mean(contact_area_left),
        "Area de contacto der (cm2)": np.mean(contact_area_right),
    }

    result = {
        "Status": 0,
        "Message": f"Frames:",
        "Result": {
            "Statistics": results,
            "PointData": [
                {
                    "Name": "Centro de presión",
                    "Values": to_Point2DIntensity(CoPs),
                    "Global": convert_nan_to_none(global_mean)
                },
                {
                    "Name": "Centro de presión izquierdo",
                    "Values": to_Point2DIntensity(CoPs_left),
                    "Global": convert_nan_to_none(global_mean_left)
                },
                {
                    "Name": "Centro de presión derecho",
                    "Values": to_Point2DIntensity(CoPs_right),
                    "Global": convert_nan_to_none(global_mean_right)
                }
            ]
        },
        "StudyID": test["ID"], 
        "AlgorithmName": "Análisis postural estático",
        "AlgorithmVersion": "1.0",
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    if not sys.stdin.isatty():
        # Hay datos en stdin, leerlos
        input_data = sys.stdin.read()
        main(input_data)
    else:
        # No hay datos en stdin, ejecución de prueba
        data = read_json_file('C:\\Users\\Angel\\AppData\\Local\\ForcePlatform\\estudios\\Estatica\\2025-07-29_21-36-48_Zeitoune_Angel_33720.json')
        main(json.dumps(data))
