from __future__ import annotations

from absl import logging
import polars as pl

from scs.utils import (
    add_cyclic_components,
    get_files_in_directory,
)

PATH_PRICES: str = (
    "data/preprocessed_data/electricity_prices/merged_electricity_prices.csv"
)
PATH_CAPACITY: str = "data/preprocessed_data/installed_capacity/renewable_share.csv"
PATH_WEATHER: str = "data/preprocessed_data/weather_data"


def merge_preprocessed_data() -> None:
    prices = pl.read_csv(PATH_PRICES)
    capacities = pl.read_csv(PATH_CAPACITY)
    prices_and_capacities = prices.join(capacities, on="timestamp", how="inner")
    logging.info(
        f"Loaded and merged:\n{PATH_PRICES}\nand\n{PATH_CAPACITY}\na"
        f"as{prices_and_capacities}"
    )
    del prices, capacities
    data_months = []
    (weather_files := get_files_in_directory(PATH_WEATHER)).sort()
    for file in weather_files:
        weather_month = pl.read_csv(file)
        data_month = prices_and_capacities.join(
            weather_month, on="timestamp", how="inner"
        )
        del weather_month
        data_month = add_cyclic_components(data_month)
        data_month = data_month.drop("POSIX_timestamp")
        data_months.append(data_month)
        logging.info(f"Loaded weather data: {file}")
    dataset = pl.concat(data_months)
    del data_months
    save_as = "data/processed_data/processed_dataset.csv"
    dataset.write_csv(save_as)
    logging.info(
        f"Successfully merged all preprocessed data into one dataset which was "
        f"saved to:\n{save_as}"
    )


def merge_preprocessed_data_monhly() -> None:
    prices = pl.read_csv(PATH_PRICES)
    capacities = pl.read_csv(PATH_CAPACITY)
    prices_and_capacities = prices.join(capacities, on="timestamp", how="inner")
    logging.info(
        f"Loaded and merged:\n{PATH_PRICES}\nand\n{PATH_CAPACITY}\na"
        f"as{prices_and_capacities}"
    )
    del prices, capacities
    (weather_files := get_files_in_directory(PATH_WEATHER)).sort()
    for file in weather_files:
        weather_month = pl.read_csv(file)
        data_month = prices_and_capacities.join(
            weather_month, on="timestamp", how="inner"
        )
        del weather_month
        data_month = add_cyclic_components(data_month)
        data_month = data_month.drop("POSIX_timestamp")
        save_as = f"data/processed_data/{file.name.split('.')[0]}_processed.csv"
        data_month.write_csv(save_as)
        logging.info(f"Merged and saved month {file.name.split('.')[0]} as: {save_as}")
