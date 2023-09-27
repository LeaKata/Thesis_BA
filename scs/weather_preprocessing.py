from __future__ import annotations

import calendar
import datetime
from multiprocessing import Pool
from typing import TYPE_CHECKING

from absl import logging
import numpy as np
import polars as pl
import xarray as xr

from scs.utils import (
    decompress_and_delete_bz2,
    delete_all_files_in_folder,
    download_and_unpack,
    gen_months_in_range,
    get_files_in_directory,
)

if TYPE_CHECKING:
    from pathlib import Path

URLS: dict[str, list[str]] = {
    "WS_080m": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_080/WS_080m.2D.",
        ".nc4",
    ],
    "WS_100m": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_100/WS_100m.2D.",
        ".nc4",
    ],
    "WS_125m": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_125/WS_125m.2D.",
        ".nc4",
    ],
    "WS_150m": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_150/WS_150m.2D.",
        ".nc4",
    ],
    "ASWDIFD_S": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIFD_S/ASWDIFD_S.2D.",
        ".grb.bz2",
    ],
    "ASWDIR_S": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIR_S/ASWDIR_S.2D.",
        ".grb.bz2",
    ],
    "ATHD_S": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/ATHD_S/ATHD_S.2D.",
        ".grb.bz2",
    ],
    "CLCT": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/CLCT/CLCT.2D.",
        ".grb.bz2",
    ],
    "T_2M": [
        "https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/T_2M/T_2M.2D.",
        ".grb.bz2",
    ],
}

MAPPING: dict[str, str] = {
    "WS_080m": "wind_speed",
    "WS_100m": "wind_speed",
    "WS_125m": "wind_speed",
    "WS_150m": "wind_speed",
    "ASWDIFD_S": "ASWDIFD_S",
    "ASWDIR_S": "ASWDIR_S",
    "ATHD_S": "ATHD_S",
    "CLCT": "tcc",
    "T_2M": "t2m",
}


def obtain_and_preprocess_weather_data(
    downsample: int,
    mask_path: str,
) -> None:
    raw_weather_directory = "data/raw_data/weather_data"
    delete_all_files_in_folder(raw_weather_directory)
    months = gen_months_in_range((2005, 12), (2019, 8))
    mask = np.load(mask_path)[::downsample, ::downsample]
    colnames = gen_weather_variable_colnames(mask)
    for month in months:
        logging.info(f"Downloading data for month: {month}")
        with Pool(9) as p:
            p.starmap(
                download_and_unpack,
                [(month.join(url), raw_weather_directory) for _, url in URLS.items()],
            )
        data = preprocess_weather_month(
            raw_weather_directory, mask, downsample, month, colnames
        )
        out_path = f"data/preprocessed_data/weather_data/{month}.csv"
        data.write_csv(out_path)
        delete_all_files_in_folder(raw_weather_directory)
        logging.info(f"Wrote preprocessed dataset for month {month} to: {out_path}")


def gen_weather_variable_colnames(mask: np.ndarray) -> dict[str, list[str]]:
    n_elements = np.sum(mask)
    colnames = {}
    for key in MAPPING:
        colnames[key] = [f"{key}_{n}" for n in range(n_elements)]
    return colnames


def preprocess_weather_month(
    directory: str,
    mask: np.ndarray,
    downsample: int,
    month: str,
    colnames: dict[str, list[str]],
) -> pl.DataFrame:
    files = get_files_in_directory(directory)
    data = {}
    for file in files:
        if file.suffix == ".bz2":
            file = decompress_and_delete_bz2(file)
        elif file.suffix == ".idx":
            continue
        logging.info(f"Processing file: {file}")
        name = file.name.split(".")[0]
        values = _read_weather_data_to_xarray(file)[MAPPING[name]].values
        # Obtain a grid of every 'downsample'th entry and store the masked
        # values as a vector in the dictionary.
        data[name] = values[:, ::downsample, ::downsample][:, mask]
        if np.any(np.isnan(data[name])):
            data[name] = np.nan_to_num(data[name])
            logging.info(f"Replaced NaN values with zeros in: {file}")
    dframes = [pl.from_dict(_gen_weather_timecodes(month))]
    dframes += [
        pl.from_numpy(data[key], schema=colnames[key], orient="row") for key in URLS
    ]
    return pl.concat(dframes, how="horizontal")


def _read_weather_data_to_xarray(filepath: Path) -> xr.Dataset:
    if filepath.suffix[:3] == ".nc":
        data = xr.open_dataset(filepath)
    elif filepath.suffix == ".grb":
        data = xr.open_dataset(filepath, engine="cfgrib")
    else:
        raise TypeError(
            f"Can only read data of type '.nc*' and '.grb'; "
            f"received: {filepath.suffix}"
        )
    return data


def _gen_weather_timecodes(year_month: str) -> dict[str, list[str]]:
    year = int(year_month[:4])
    month = int(year_month[4:])
    start = datetime.datetime(year, month, 1, 0, 0)
    # Determine the number of days in the month
    _, num_days = calendar.monthrange(year, month)
    num_hours = num_days * 24
    timestamps = [start + datetime.timedelta(hours=i) for i in range(num_hours)]
    return {
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M+00:00") for t in timestamps],
    }
