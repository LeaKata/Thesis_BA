from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scs.weather_preprocessing import (
    MAPPING,
    _read_weather_data_to_xarray,
)

data = _read_weather_data_to_xarray(
    Path("data/raw_data/weather_data/T_2M.2D.201908.grb")
)
data = data["t2m"].values

mask = np.load("data/utils/mask.npy")


def save_to_csv_and_image(array: np.ndarray, filename: str) -> None:
    # Ensure the array is a numpy array
    array = np.flip(array, 0)

    # First CSV: row indices as column names
    with open(filename + ".csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + list(range(array.shape[1])))  # Column names
        for i, row in enumerate(array):
            writer.writerow([i] + list(row))

    # Second CSV: X, Y, Z format
    with open(filename + "_cords.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])  # Column names
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                writer.writerow([i, j, array[i, j]])

    # Save as image
    plt.imshow(array, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


for name, ext in [
    ("ASWDIR_S", "grb"),
    ("CLCT", "grb"),
    ("WS_100m", "nc4"),
    ("T_2M", "grb"),
]:
    data = _read_weather_data_to_xarray(
        Path(f"data/raw_data/weather_data/{name}.2D.201908.{ext}")
    )
    data = data[MAPPING[f"{name}"]].values
    save_to_csv_and_image(data[106], f"{name}_106")
