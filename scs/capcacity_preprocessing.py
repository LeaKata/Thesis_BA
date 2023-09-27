from __future__ import annotations

from absl import logging
import polars as pl

from scs.utils import gen_hours_in_range


def preprocess_capacity_data(filepath: str) -> None:
    capacity_data = pl.read_csv(filepath, skip_rows_after_header=1)
    logging.info(f"Preprocessing capcacity data: {filepath}")
    capacity_data = _translate_capacity_colnames(capacity_data)
    capacity_data = _get_renewable_shares(capacity_data)
    timerange = gen_hours_in_range((2005, 12, 31, 23), (2023, 6, 22, 21))
    hourly_data = pl.from_dict(timerange)
    # Add column just containing the year
    hourly_data = hourly_data.with_columns(
        pl.col("timestamp").str.slice(0, 4).cast(pl.Int64).alias("year")
    )
    capacity_data = capacity_data[["year", "wind_share", "solar_share"]]
    hourly_data = hourly_data.join(capacity_data, on="year", how="inner")
    hourly_data = hourly_data.drop("year")
    save_as = "data/preprocessed_data/installed_capacity/renewable_share.csv"
    hourly_data.write_csv(save_as, separator=",")
    logging.info(f"Wrote preprocessed capacity data to: {save_as}")


def _translate_capacity_colnames(df: pl.DataFrame) -> pl.DataFrame:
    german_columns = df.columns
    df = df.with_columns(
        [
            pl.col("Jahr").alias("year"),
            pl.col("Kernenergie").alias("nuclear_energy"),
            pl.col("Braunkohle").alias("lignite"),
            pl.col("Steinkohle").alias("hard_coal"),
            pl.col("Erdgas").alias("natural_gas"),
            pl.col("Mineralöl").alias("mineral_oil"),
            pl.col("Laufwasser").alias("running_water"),
            pl.col("Biomasse").alias("biomass"),
            pl.col("Wind onshore").alias("wind_onshore"),
            pl.col("Wind offshore").alias("wind_offshore"),
            pl.col("Solar").alias("solar"),
        ]
    )
    return df.drop(german_columns)


def _get_renewable_shares(df: pl.DataFrame) -> pl.DataFrame:
    # Sum over all columns to get total capacity
    df = df.with_columns(
        pl.sum([pl.col(c) for c in df.columns[1:]]).alias("total_capacity")
    )
    df = df.with_columns(
        (pl.col("wind_onshore") + pl.col("wind_offshore")).alias("total_wind")
    )
    df = df.with_columns(
        (pl.col("total_wind") / pl.col("total_capacity")).alias("wind_share")
    )
    df = df.with_columns(
        (pl.col("solar") / pl.col("total_capacity")).alias("solar_share")
    )
    return df
