from __future__ import annotations

from absl import logging
import polars as pl
import tensorflow as tf

from scs.data_windowing import TSDatasetGenerator
from scs.utils import (
    autoregressive_evaluation,
    gen_warmup_window,
    generate_comparison,
    plot_predictions,
    print_metrics,
    save_predictions,
)

# tf.config.set_visible_devices([], "GPU")
# tf.random.set_seed(1234)
# tf.config.run_functions_eagerly(True)  # for debugging


logging.set_verbosity(logging.INFO)

data = pl.read_csv("data/processed_data/processed_dataset_50.csv")

# Config data windowing here:
WINDOW_SIZE: int = 6
TSDG: TSDatasetGenerator = TSDatasetGenerator(WINDOW_SIZE, 1, input_offset=-1)
DATA_INPUT: list[str] = data.columns[1:]
DATA_TARGET: list[str] = ["day_ahead_auction"]
BATCHSIZE: int = 128

TSDG.link_dataframe(data, DATA_INPUT, DATA_TARGET)

train_dataset, test_dataset, eval_dataset_df = TSDG.split_data_train_test_eval(
    test_eval_ratios=(0.1, 0.01), normalize=False
)
train_dataset = TSDG.gen_tf_dataset(BATCHSIZE, train_dataset)
test_dataset = TSDG.gen_tf_dataset(BATCHSIZE, test_dataset)
eval_dataset = TSDG.gen_tf_dataset(1, eval_dataset_df)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(6, 67)),
        tf.keras.layers.Dense(units=2048, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1),
    ]
)


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
)

history = model.fit(
    train_dataset,
    epochs=4,
    shuffle=True,
)


history = model.evaluate(test_dataset)

warmup = gen_warmup_window(eval_dataset_df, "day_ahead_auction", WINDOW_SIZE, 48)
warmup_timesteps = 48 - WINDOW_SIZE
test_true, test_pred = generate_comparison(model, test_dataset)
plot_predictions(test_true, test_pred)

eval_true, eval_pred, warmup = autoregressive_evaluation(
    model, eval_dataset, warmup_timesteps, WINDOW_SIZE, warmup_input=warmup
)
print_metrics(eval_true, eval_pred)
plot_predictions(eval_true, eval_pred, warmup=warmup)
save_predictions(eval_pred, "experiment_3", 250, 48)
