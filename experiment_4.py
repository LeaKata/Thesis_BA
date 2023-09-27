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
WINDOW_SIZE: int = 13
OFFSET: int = 7
TSDG: TSDatasetGenerator = TSDatasetGenerator(WINDOW_SIZE, 1, input_offset=OFFSET)
DATA_INPUT: list[str] = data.columns[1:]
DATA_TARGET: list[str] = ["day_ahead_auction"]
BATCHSIZE: int = 500

TSDG.link_dataframe(data, DATA_INPUT, DATA_TARGET)

train_dataset, test_dataset, eval_dataset_df = TSDG.split_data_train_test_eval(
    test_eval_ratios=(0.1, 0.01), normalize=True
)
train_dataset = TSDG.gen_tf_dataset(BATCHSIZE, train_dataset)
test_dataset = TSDG.gen_tf_dataset(BATCHSIZE, test_dataset)
eval_dataset = TSDG.gen_tf_dataset(1, eval_dataset_df)


# Input Data Head
id_input = tf.keras.layers.Input(shape=(WINDOW_SIZE, 66))
id_flatten = tf.keras.layers.Flatten()(id_input)
id_dense = tf.keras.layers.Dense(units=1024, activation="relu")(id_flatten)

# Autoregressive Head
ar_input = tf.keras.layers.Input(shape=(WINDOW_SIZE - (OFFSET + 1), 1))
ar_flatten = tf.keras.layers.Flatten()(ar_input)
ar_dense = tf.keras.layers.Dense(units=128, activation="relu")(ar_flatten)

# Dense Tower
dt_input = tf.keras.layers.concatenate([id_dense, ar_dense])
dt_dense_1 = tf.keras.layers.Dense(units=2048, activation="relu")(dt_input)
dt_dropout_1 = tf.keras.layers.Dropout(0.2)(dt_dense_1)
dt_dense_2 = tf.keras.layers.Dense(units=1024, activation="relu")(dt_dropout_1)
dt_dropout_2 = tf.keras.layers.Dropout(0.2)(dt_dense_2)
dt_dense_3 = tf.keras.layers.Dense(units=512, activation="relu")(dt_dropout_2)
dt_dropout_3 = tf.keras.layers.Dropout(0.2)(dt_dense_3)
dt_dense_4 = tf.keras.layers.Dense(units=256, activation="relu")(dt_dropout_3)
dt_dropout_4 = tf.keras.layers.Dropout(0.2)(dt_dense_4)
out_layer = tf.keras.layers.Dense(units=1)(dt_dropout_4)

model = tf.keras.Model(inputs=[id_input, ar_input], outputs=out_layer)


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
)

history = model.fit(
    train_dataset,
    epochs=7,
    shuffle=True,
)

history = model.evaluate(test_dataset)

warmup = gen_warmup_window(eval_dataset_df, "day_ahead_auction", WINDOW_SIZE, 48)
warmup_timesteps = 48 - WINDOW_SIZE
test_true, test_pred = generate_comparison(model, test_dataset)
plot_predictions(test_true, test_pred)

eval_true, eval_pred, warmup = autoregressive_evaluation(
    model,
    eval_dataset,
    warmup_timesteps,
    WINDOW_SIZE - (OFFSET + 1),
    warmup_input=warmup,
)
print_metrics(eval_true, eval_pred)
plot_predictions(eval_true, eval_pred, warmup=warmup)
save_predictions(eval_pred, "experiment_4", 250, 48)
