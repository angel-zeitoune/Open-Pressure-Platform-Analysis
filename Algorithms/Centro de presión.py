""" Cálculo del centro de presión global de cada frame. """

import sys
import json

from core.decode_matrix import decode_matrix
from formats.formats import convert_nan_to_none, to_Point2DIntensity
from func.calculate_center_of_pressure import calculate_center_of_pressure_video


def main():
    # cargar archivo de prueba "C:\\Users\\Angel\\AppData\\Local\\ForcePlatform\\algoritmos\\Centro de presión.py"
    # with open("C:\\Users\\Angel\\AppData\\Local\\ForcePlatform\\estudios\\Idan\\2025-05-25_16-59-01_Zeitoune_Idan_1760.json", "r") as file:
    #     test = json.load(file)

    input_data = sys.stdin.read()
    test = json.loads(input_data)

    rows = test["Platform"]["Rows"]
    cols = test["Platform"]["Columns"]
    frames = test["Platform"]["Frames"]
    total_duration = test.get("Duration", 1) / 1000 # en segundos

    data = [decode_matrix(frame["Data"], rows, cols) for frame in frames]
    CoPs, global_mean, traj_length, traj_mean = calculate_center_of_pressure_video(data)
    mean_velocity = traj_length / total_duration

    result = {
        "Status": 0,
        "Message": f"Frames: {len(frames)}",
        "Result": {
            "PointData": [
                {
                    "Name": "Centro de presión",
                    "Values": to_Point2DIntensity(CoPs), # Convertir los resultados a un formato compatible
                    "Global": convert_nan_to_none(global_mean)
                }
            ],
            "Statistics": {
                "Trayectoria total": convert_nan_to_none(traj_length),
                "Trayectoria media": convert_nan_to_none(traj_mean),
                "Velocidad media": convert_nan_to_none(mean_velocity)
            }

        },
        "StudyID": test["ID"], 
        "AlgorithmName": "Test",
        "AlgorithmVersion": "1.0",
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
