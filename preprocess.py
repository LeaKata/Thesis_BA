from __future__ import annotations

from absl import logging

from scs.capcacity_preprocessing import preprocess_capacity_data
from scs.preprocess_dataset import merge_preprocessed_data
from scs.prices_preprocessing import merge_and_preprocess_electicity_price_data
from scs.weather_preprocessing import obtain_and_preprocess_weather_data

logging.set_verbosity(logging.INFO)
merge_and_preprocess_electicity_price_data()
preprocess_capacity_data(
    "data/raw_data/installed_capacity/energy-charts_Installierte_Netto-Leistung_zur_Stromerzeugung_in_Deutschland.csv"
)
obtain_and_preprocess_weather_data(downsample=50, mask_path="data/utils/mask.npy")
merge_preprocessed_data()
