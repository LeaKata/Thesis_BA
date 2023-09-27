from __future__ import annotations

from typing import TYPE_CHECKING

from absl import logging
import numpy as np
import polars as pl
import tensorflow as tf

if TYPE_CHECKING:
    from typing import Iterator


class TSDatasetGenerator:
    input_indices: np.ndarray
    target_indices: np.ndarray
    total_window_size: int
    data: pl.DataFrame
    input_columns: list[str]
    target_columns: list[str]
    columns: list[str]
    autoregressive_input_target_overlap: list[str]
    target_overlap: bool
    n_input: int
    n_target: int
    ar_input_target_overlap: int

    def __init__(
        self,
        input_width: int,
        target_width: int,
        input_offset: int = -1,
    ) -> None:
        """Initializes the TSDatasetGenerator with given input and target widths.

        Key Points:
        - All indices are relative to the prediction time point.
        - Indices can be adjusted using the `input_offset`.

        Examples:
        - Input width: 4, Target width: 1
            * Input offset: -1
            - Predicts target data at index [0] using indices [-4, -3, -2, -1].
            - e.g., Predicting tomorrow's price based on the last 4 days of weather,
              including today.

            * Input offset: 0
            - Predicts target data at index [0] using indices [-3, -2, -1, 0].
            - e.g., Predicting tomorrow's price using the last 3 days of weather
              and tomorrow's forecast.

            * Input offset: 2
            - Predicts target data at index [0] using indices [-1, 0, 1, 2].
            - e.g., Using data from yesterday, today, tomorrow, and the day after
              for predictions. Suitable when we have reliable future input data models.

        Notes:
        - `target_width` determines how many future timesteps we want to predict.
            - Width of 1: Predicts data at index [0]
            - Width of 3: Predicts data at indices [0, 1, 2].

        Args:
            input_width (int):
                The width of the input window starting at 1 and looking backwards.

            target_width (int):
                The width of the target window starting at 0 and looking forwards.

            input_offset (int, optional):
                The offset of the input window relative to the target index [0].
                Defaults to -1.
        """
        self.input_width: int = input_width
        self.target_width: int = target_width
        # Need +1 since python counts to up to n-1 when slicing
        self.input_offset: int = input_offset + 1
        self._gen_window_inidices()
        logging.info(f"Initialized WindowGenerator with:\n{self}")

    def __str__(self) -> str:
        return "\n".join(
            [
                f"total window size: {self.total_window_size}",
                f"input indices: {self.input_indices}",
                f"input_offset: {self.input_offset - 1}",
                f"target indices: {self.target_indices}",
            ]
        )

    def _gen_window_inidices(self) -> None:
        self.input_indices = np.arange(
            -self.input_width + self.input_offset, self.input_offset
        )
        self.target_indices = np.arange(self.target_width)
        all_indices = np.concatenate((self.input_indices, self.target_indices))
        min_index = np.min(all_indices)
        max_index = np.max(all_indices)
        self.total_window_size = max_index - min_index + 1
        self.input_indices -= min_index
        self.target_indices -= min_index

    def _validate_window(self, autoregressive_columns: list[str]) -> None:
        """
        Ensures valid window shapes based on autoregressive input components.

        When there's an overlap of columns between the input and target (indicative
        of an autoregressive component), this method checks the validity of window
        shapes. The principle is that using current or future values to predict
        the current value is not valid. For predicting a value at time `t`, only
        past values (up to `t-1`) should be used for autoregressive inputs.
        Non-autoregressive inputs don't have such restrictions, therefore no such
        checks are required.

        Args:
            autoregressive_columns (list[str]): Columns shared between input and target.

        Raises:
            ValueError: If the use of autoregressive input columns isn't appropriate.
        """
        if not autoregressive_columns:
            logging.info(
                "No common input and target columns; All window shapes are valid."
            )
            return
        if self.input_offset >= self.input_width:
            raise ValueError(
                f"Can not use autoregressive input {autoregressive_columns} "
                f"with values from t >= 0 only."
            )

    def link_dataframe(
        self,
        data: pl.DataFrame,
        input_columns: list[str],
        target_columns: list[str],
    ) -> None:
        """Links the provided dataframe with the generator.

        Key Points:
        - Establishes specifications for input, target, and overlap.

        Definitions:
        - Autoregressive Columns: Columns shared between input and target sets.
        - AR Input Target Overlap: Autoregressive Columns where input and target
          windows temporally intersect due to the offset and therefore must be
          adjusted to the available data at each timestep.
        """
        self.data = data
        self.target_columns = target_columns
        target_set = set(target_columns)
        input_set = set(input_columns)
        autoregressive_columns = [
            colname for colname in target_columns if colname in input_set
        ]
        self._validate_window(autoregressive_columns)
        self.input_columns = [
            colname for colname in input_columns if colname not in target_set
        ]
        # Sort complete list of columns: exclusive input columns first, autoregressive
        # columns second, exclusive target columns third.
        self.columns = (
            self.input_columns
            + autoregressive_columns
            + [colname for colname in target_columns if colname not in input_set]
        )
        self.process_with_overlap = self.input_offset >= 1 and any(
            autoregressive_columns
        )
        if not self.process_with_overlap:
            # If no overlapping windows, append autoregressive columns since no
            # distinction between input and autoregressive columns is needed.
            self.input_columns += autoregressive_columns
            self.autoregressive_input_target_overlap = []
        else:
            self.autoregressive_input_target_overlap = autoregressive_columns
        self.n_input = len(self.input_columns)
        self.ar_input_target_overlap = len(self.autoregressive_input_target_overlap)
        self.n_target = len(self.target_columns)

    def normalize_dataframe(
        self, data: pl.DataFrame, realtive_to: pl.DataFrame
    ) -> pl.DataFrame:
        mean = realtive_to.mean()
        std = realtive_to.std()
        return data.with_columns(
            (pl.col(colname) - mean[colname][0]) / std[colname][0]
            for colname in data.columns
        )

    def denormalize(self, predictions: list[float], train_ration: float) -> list[float]:
        if len(self.target_columns) > 1:
            raise ValueError("This method only works for single prediction variables")
        n = self.data.shape[0]
        train_size = int(train_ration * n)
        train_targets = self.data[self.target_columns][:train_size]
        mean = train_targets.mean().item()
        std = train_targets.std().item()
        return [pred * std + mean for pred in predictions]

    def split_data_train_test_eval(
        self,
        test_eval_ratios: tuple[float, float],
        normalize: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        n = self.data.shape[0]
        test_size = int(test_eval_ratios[0] * n)
        eval_size = int(test_eval_ratios[1] * n)
        train_size = n - test_size - eval_size
        train_data = self.data[self.columns][:train_size]
        test_data = self.data[self.columns][train_size : train_size + test_size]
        eval_data = self.data[self.columns][train_size + test_size :]
        if normalize:
            train_data = self.normalize_dataframe(train_data, train_data)
            if test_data.shape[0] > 0:
                test_data = self.normalize_dataframe(test_data, train_data)
            eval_data = self.normalize_dataframe(eval_data, train_data)
        return train_data, test_data, eval_data

    def _split_window(self, window: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        input_data = tf.gather(window, self.input_indices)[:, : self.n_input]
        target_data = tf.gather(window, self.target_indices)[:, -self.n_target :]
        return input_data, target_data

    def _split_window_ar_overlap(
        self, window: tf.Tensor
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        input_data = tf.gather(window, self.input_indices)[:, : self.n_input]
        target_data = tf.gather(window, self.target_indices)[:, -self.n_target :]
        autoregressive_input_data = tf.gather(
            window, self.input_indices[: -self.input_offset]
        )
        autoregressive_input_data = autoregressive_input_data[
            :, self.n_input : self.n_input + self.ar_input_target_overlap
        ]
        return (input_data, autoregressive_input_data), target_data

    def gen_tf_dataset(
        self, batchsize: int | None, data: pl.DataFrame | None = None
    ) -> tf.data.Dataset:
        if not hasattr(self, "data"):
            raise ValueError("No data available; linking of data required.")
        if data is None:
            data = self.data
        dataset = tf.data.Dataset.from_tensor_slices(data[self.columns].to_numpy())
        dataset = dataset.window(self.total_window_size, shift=1, drop_remainder=True)
        cardinality = dataset.cardinality()
        dataset = dataset.flat_map(lambda window: window.batch(self.total_window_size))
        # Setting cardinality manually due to variable lengths from ".flat_map()"
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
        if self.process_with_overlap:
            dataset = dataset.map(self._split_window_ar_overlap)
        else:
            dataset = dataset.map(self._split_window)
        if batchsize:
            dataset = dataset.batch(batchsize, drop_remainder=True)
        return dataset

    def cross_validation_iterator(
        self,
        data: pl.DataFrame,
        k_folds: int,
        tt_split: float,
        normalize: bool,
        batchsize: int | None = None,
    ) -> Iterator[tuple[tf.data.Dataset, tf.data.Dataset]]:
        n = data.shape[0]
        train_size = int(tt_split * n)
        test_size = n - train_size
        for k in range(k_folds):
            train_data = pl.concat([data[: k * test_size], data[(k + 1) * test_size :]])
            test_data = data[k * test_size : (k + 1) * test_size]
            if normalize:
                train_data = self.normalize_dataframe(train_data, train_data)
                test_data = self.normalize_dataframe(test_data, train_data)
            yield (
                self.gen_tf_dataset(batchsize, train_data),
                self.gen_tf_dataset(batchsize, test_data),
            )
