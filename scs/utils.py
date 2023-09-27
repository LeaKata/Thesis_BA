from __future__ import annotations

import bz2
from collections import deque
import datetime
from pathlib import Path
import shutil

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import tensorflow as tf
from tqdm import tqdm


def get_files_in_directory(directory: str) -> list[Path]:
    if not isinstance(directory, str):
        raise TypeError("Input must be a string.")
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError("Directory does not exist.")
    return [file for file in directory_path.iterdir() if file.is_file()]


def delete_all_files_in_folder(directory: str) -> None:
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"'{directory}' is not a directory.")
    for file in path.glob("*"):
        if file.is_file():
            file.unlink()


def download_and_unpack(url: str, save_to: str) -> None:
    """Download a file from a URL and return the path to the file.

    This function sends a HTTP GET request to the specified URL and writes
    the response content to a file. The file name is determined from the last
    part of the URL. If the request or the file writing operation fails, the
    function returns None.
    """
    logging.info(f"Attempting to download: {url}")
    try:
        response = requests.get(url, stream=True)
        # Check if the request was successful
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.info(f"Failed to download file: {e}")
        return None

    filename = url.split("/")[-1]
    path = Path(save_to) / filename

    try:
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        logging.info(f"Failed to save file: {e}")
        return None
    if path.suffix == ".bz2":
        decompress_and_delete_bz2(path)


def decompress_and_delete_bz2(filepath: Path) -> Path:
    """Decompress a .bz2 file and delete the original compressed file.

    This function takes as input a pathlib.Path object that points to a .bz2 file,
    decompresses the file, and saves the output in a new file with the same name
    (but without the .bz2 extension). The original .bz2 file is then deleted.
    """
    new_filepath = filepath.with_suffix("")
    # Decompress the bz2 file
    with bz2.BZ2File(filepath, "rb") as f_in:
        with new_filepath.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # Delete the original bz2 file
    filepath.unlink()
    logging.info(f"Successfully decompressed and deleted: {filepath}")
    return new_filepath


def gen_months_in_range(start: tuple[int, int], end: tuple[int, int]) -> list[str]:
    start_year, start_month = start
    end_year, end_month = end
    current_year, current_month = start_year, start_month
    months = []
    while (current_year, current_month) <= (end_year, end_month):
        months.append(f"{current_year:04d}{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_year += 1
            current_month = 1
    return months


def gen_hours_in_range(
    start: tuple[int, int, int, int], end: tuple[int, int, int, int]
) -> dict[str, list[str]]:
    start_year, start_month, start_day, start_hour = start
    end_year, end_month, end_day, end_hour = end
    start_time = datetime.datetime(start_year, start_month, start_day, start_hour)
    end_time = datetime.datetime(end_year, end_month, end_day, end_hour)
    # Calculate the total number of hours between the start and end time
    num_hours = int((end_time - start_time).total_seconds() // 3600)
    hours = [start_time + datetime.timedelta(hours=i) for i in range(num_hours + 1)]
    return {"timestamp": [h.strftime("%Y-%m-%dT%H:%M+00:00") for h in hours]}


def add_cyclic_components(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds the cyclic components as identified via Fourier transform.
    """
    df = df.with_columns(
        pl.col("timestamp")
        .str.to_datetime("%Y-%m-%dT%H:%M%z")
        .dt.timestamp()
        .alias("POSIX_timestamp")
    )
    df_ts = df["POSIX_timestamp"]
    day = 24 * 60 * 60
    year = 365.2425 * day
    month = year / 12
    week = year / 52.1429
    half_day = day / 2

    df = df.with_columns((np.sin(df_ts * (2 * np.pi / year))).alias("year_sin"))  # type: ignore[attr-defined]
    df = df.with_columns((np.cos(df_ts * (2 * np.pi / year))).alias("year_cos"))  # type: ignore[attr-defined]

    df = df.with_columns((np.sin(df_ts * (2 * np.pi / month))).alias("month_sin"))  # type: ignore[attr-defined]
    df = df.with_columns((np.cos(df_ts * (2 * np.pi / month))).alias("month_cos"))  # type: ignore[attr-defined]

    df = df.with_columns((np.sin(df_ts * (2 * np.pi / week))).alias("week_sin"))  # type: ignore[attr-defined]
    df = df.with_columns((np.cos(df_ts * (2 * np.pi / week))).alias("week_cos"))  # type: ignore[attr-defined]

    df = df.with_columns((np.sin(df_ts * (2 * np.pi / day))).alias("day_sin"))  # type: ignore[attr-defined]
    df = df.with_columns((np.cos(df_ts * (2 * np.pi / day))).alias("day_cos"))  # type: ignore[attr-defined]

    df = df.with_columns((np.sin(df_ts * (2 * np.pi / half_day))).alias("12h_sin"))  # type: ignore[attr-defined]
    df = df.with_columns((np.cos(df_ts * (2 * np.pi / half_day))).alias("12h_cos"))  # type: ignore[attr-defined]

    return df


def generate_comparison(
    model: tf.keras.Model, dataset: tf.data.Dataset, start_at: int = 0
) -> tuple[list[float], list[float]]:
    true_values = []
    predictions = []
    for i, d in tqdm(enumerate(dataset), total=dataset.cardinality().numpy()):
        if i < start_at:
            continue
        true_value = tf.squeeze(d[1]).numpy()
        prediction = tf.squeeze(model(d[0], training=False)).numpy()
        if isinstance(true_value, np.ndarray):
            true_values += true_value.tolist()
            predictions += prediction.tolist()
        else:
            true_values.append(true_value)
            predictions.append(prediction)
    return true_values, predictions


def gen_warmup_window(
    data: pl.DataFrame, colunmname: str, window_size: int, start_at: int
) -> list[float]:
    return data[colunmname][start_at - window_size : start_at].to_list()


def autoregressive_evaluation(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    start_at: int,
    window_size: int,
    max_ar_steps: None | int = None,
    warmup_input: None | list[float] = None,
) -> tuple[list[float], list[float], list[float]]:
    """Evaluates an autoregressive model using given data.

    This function will iterate over the dataset and evaluate the provided model in an
    autoregressive manner. If warmup input is provided, it will be used as initial
    data; otherwise, the function uses the early dataset samples for this purpose.

    Args:
        model (tf.keras.Model): The autoregressive model to evaluate.
        dataset (tf.data.Dataset): The dataset to evaluate against.
        start_at (int): The index from which actual evaluation begins.
        window_size (int): The number of steps/samples to warm up.
        warmup_input (list[float], optional): Initial data to warm up. Defaults to None.
    """
    true_values = []
    warmup = []
    predictions = []

    if max_ar_steps is None:
        max_ar_steps = window_size

    if warmup_input is not None:
        ar_input = deque(warmup_input, maxlen=max_ar_steps)
    else:
        ar_input = deque(maxlen=max_ar_steps)
        warmup = list(ar_input)

    for i, d in tqdm(enumerate(dataset), total=dataset.cardinality().numpy()):
        current_true_value = tf.squeeze(d[1]).numpy()
        if i < start_at:
            if warmup_input is None:
                ar_input.append(current_true_value)
                warmup.append(current_true_value)
            continue
        true_values.append(current_true_value)
        if max_ar_steps < window_size:
            merge_inputs = true_values[-window_size:].copy()
            if max_ar_steps > 0:
                merge_inputs[-max_ar_steps:] = list(ar_input)
        else:
            merge_inputs = list(ar_input)
        model_input = _gen_ar_model_input(d[0], ar_input, window_size)
        prediction = tf.squeeze(model(model_input, training=False)).numpy()
        ar_input.append(prediction)
        predictions.append(prediction)

    return true_values, predictions, warmup


def _gen_ar_model_input(
    data_t: tf.Tensor | tuple[tf.Tensor, tf.Tensor],
    ar_input: deque[float],
    ar_window_size: int,
) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
    ar_data = tf.constant([[ar_input]], dtype=tf.float64)
    ar_data = tf.reshape(ar_data, (1, ar_window_size, 1))
    if isinstance(data_t, tuple):  # Compare with input for autoregressive models
        return (data_t[0], ar_data)
    else:
        return tf.concat([data_t[..., :-1], ar_data], axis=2)


def save_predictions(
    predictions: list[float], experiment: str, max_timesteps: int, start_at: int = 0
) -> None:
    save_path = f"data/viz/{experiment}.csv"
    pl.DataFrame(
        {
            "index": np.arange(start_at, max_timesteps),
            f"{experiment}_predictions": predictions[: max_timesteps - start_at],
        }
    ).write_csv(save_path)


def plot_predictions(
    true_values: list[float],
    pred_values: list[float],
    warmup: None | list[float] = None,
    title: None | str = None,
    save_to: None | str = None,
    range: None | tuple[int, int] = None,
) -> None:
    _, ax = plt.subplots(figsize=(30, 10))
    n = np.arange(len(true_values))
    if warmup is not None:
        warmup.append(true_values[0])  # Add first true value to connect the plots
        n += len(warmup) - 1
        n_warmup = np.arange(len(warmup))

    if range:
        n = n[range[0] : range[1]]
        if warmup and range[0] < len(warmup):
            ax.plot(n_warmup[range[0] :], warmup[range[0] :], label="Warmup", c="g")
        ax.plot(n, true_values[range[0] : range[1]], label="True", c="b")
        ax.plot(n, pred_values[range[0] : range[1]], label="Predicted", c="r")
    else:
        if warmup is not None:
            ax.plot(n_warmup, warmup, label="Warmup", c="g")
        ax.plot(n, true_values, label="True", c="b")
        ax.plot(n, pred_values, label="Predicted", c="r")
    if title:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel("Time (h)", fontsize=16)
    ax.set_ylabel("Price (€/MWh)", fontsize=16)
    ax.legend(fontsize=16)

    if save_to:
        plt.savefig(save_to)
        plt.close()
        logging.info(f"Saved plot to: {save_to}")
    else:
        plt.show()


def print_metrics(true_values: list[float], pred_values: list[float]) -> None:
    abs_difference = [abs(true - pred) for true, pred in zip(true_values, pred_values)]
    square_difference = [abs_diff**2 for abs_diff in abs_difference]
    print(
        f"Mean Absolute Error: {np.mean(abs_difference):.2f}\n"
        f"Mean Squared Error: {np.mean(square_difference):.2f}"
    )
