from __future__ import annotations

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
# tf.config.run_functions_eagerly(True)  # for debugging

logging.set_verbosity(logging.DEBUG)

df = pl.read_csv(
    "data/preprocessed_data/electricity_prices/merged_electricity_prices.csv"
)


fft = tf.signal.rfft(df["day_ahead_auction"])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df["day_ahead_auction"])
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / (hours_per_year)

f_per_year = f_per_dataset / years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale("log")
plt.ylim(0, 400000)
# plt.xlim([0.1, max(plt.xlim())])
plt.xticks(
    [1, 12, 52.1789, 365.2524, 365.2524 * 2],
    labels=["1/year", "1/month", "1/week", "1/day", "1/0.5dayh"],
)
_ = plt.xlabel("Frequency (log scale)")
plt.show()


df = pl.DataFrame(
    {
        "fft_prices": np.abs(fft.numpy()),
        "freq_per_dataset": f_per_dataset,
    }
)
df.write_csv("data/utils/fft_prices.csv")
