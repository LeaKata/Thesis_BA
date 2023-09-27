from __future__ import annotations

from pathlib import Path

from absl import logging
import polars as pl


def merge_and_preprocess_electicity_price_data(
    data_directory: str = "data/raw_data/electricity_prices",
) -> None:
    (pathlist := list(Path(data_directory).glob("*.csv"))).sort()
    dataframes = []
    collect_columns = ["Date (UTC)", "Day Ahead Auction"]
    logging.info(
        f"Reading data from: {data_directory}/**\n"
        f"Collecting columns: {collect_columns}"
    )
    for filepath in pathlist:
        df = pl.read_csv(
            filepath,
            skip_rows_after_header=1,
        )
        if "Day Ahead Auction" not in df.columns:
            df = _merge_transition_period(df, filepath)
        dataframes.append(df[collect_columns])
        logging.info(
            f"Collected data from {df['Date (UTC)'][0]} to {df['Date (UTC)'][-1]}"
        )
    merged_data = pl.concat(dataframes)
    del dataframes
    merged_data = merged_data.rename(
        {
            "Date (UTC)": "timestamp",
            "Day Ahead Auction": "day_ahead_auction",
        }
    )
    merged_data = _handle_missing_price_observations(merged_data)
    save_as = "data/preprocessed_data/electricity_prices/merged_electricity_prices.csv"
    merged_data.write_csv(save_as, separator=",")
    logging.info(f"Electricity price data merged and saved to: {save_as}")


def _merge_transition_period(df: pl.DataFrame, filepath: Path) -> pl.DataFrame:
    """
    Merges the column 'Day Ahead Auction (DE-AT-LU)' with the column
    'Day Ahead Auction (DE-LU)' as the aggregated pricing area changed in 2018.
    """
    logging.info(
        f"Accounting for the 'Day Ahead Auction (DE-AT-LU)' to "
        f"'Day Ahead Auction (DE-LU)' transition in: {filepath}"
    )
    return df.with_columns(
        pl.when(pl.col("Day Ahead Auction (DE-AT-LU)").is_null())
        .then(pl.col("Day Ahead Auction (DE-LU)"))
        .otherwise(pl.col("Day Ahead Auction (DE-AT-LU)"))
        .cast(pl.Float64)
        .alias("Day Ahead Auction")
    )


def _handle_missing_price_observations(df: pl.DataFrame) -> pl.DataFrame:
    """Fill with average?"""
    logging.info(f"Handling missing observations: {df.null_count()}")
    return df.interpolate()
