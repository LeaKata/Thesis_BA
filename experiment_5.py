from __future__ import annotations

from absl import logging
import polars as pl
import tensorflow as tf

from scs.data_windowing import TSDatasetGenerator
from scs.utils import (
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
WINDOW_SIZE: int = 12
OFFSET: int = -1
TSDG: TSDatasetGenerator = TSDatasetGenerator(WINDOW_SIZE, 1, input_offset=OFFSET)
DATA_INPUT: list[str] = data.columns[2:]
DATA_TARGET: list[str] = ["day_ahead_auction"]
BATCHSIZE: int = 4

TSDG.link_dataframe(data, DATA_INPUT, DATA_TARGET)

train_dataset, test_dataset, eval_dataset = TSDG.split_data_train_test_eval(
    test_eval_ratios=(0.1, 0.01), normalize=False
)
train_dataset = TSDG.gen_tf_dataset(BATCHSIZE, train_dataset)
test_dataset = TSDG.gen_tf_dataset(BATCHSIZE, test_dataset)
eval_dataset = TSDG.gen_tf_dataset(1, eval_dataset)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(WINDOW_SIZE, 66), batch_size=BATCHSIZE),
        tf.keras.layers.LSTM(units=512),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=1),
    ]
)


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
)

history = model.fit(
    train_dataset,
    epochs=2,
    shuffle=False,
)

history = model.evaluate(test_dataset)

warmup_timesteps = 48 - WINDOW_SIZE
test_true, test_pred = generate_comparison(model, test_dataset)
plot_predictions(test_true, test_pred)

eval_true, eval_pred = generate_comparison(
    model, eval_dataset, start_at=warmup_timesteps
)
print_metrics(eval_true, eval_pred)
plot_predictions(eval_true, eval_pred)
save_predictions(eval_pred, "experiment_5", 250, 48)
