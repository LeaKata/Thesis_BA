# Bachelor Thesis Project

For the description of the contents, see the thesis paper, this readme only explains how to run the experiments.


## Setup

### Environment

Create a python virtual environment for with `python3.11` and install the required dependencies:

```bash
$ python3.11 -m venv .ba_venv
$ source .ba_venv/bin/activate
(.ba_venv)$ pip install -r requirements.txt
```

If you want to use GPU accelerated Tensorflow make sure you install the required [CUDA](https://developer.nvidia.com/cuda-toolkit) and [CUDNN](https://docs.nvidia.com/cudnn/) [dependencies](https://www.tensorflow.org/install/source#tested_build_configurations) and configure the required paths.

Each experiment file has this section at the top
```python
# tf.config.set_visible_devices([], "GPU")
# tf.random.set_seed(1234)
# tf.config.run_functions_eagerly(True)  # for debugging```
```
to configure the GPU usage, random seed, and eager execution.
Uncomment the lines to enable the respective feature (e.g. if you do not want to use GPU accelerated tensorflow, uncomment the first line).

### Data

Either copy the `data` folder with structure

```
[project]$
├── data
│   ├── preprocessed_data
│   │   ├── electricity_prices
│   │   ├── installed_capacity
│   │   ├── weather_data
│   ├── processed_data
│   ├── raw_data
│   │   ├── electricity_prices
│   │   ├── installed_capacity
│   │   ├── weather_data
│   ├── stats
│   │   ├── yearly_day_ahead
│   ├── utils
│   ├── viz

```

as provided with the submission in the root folder of the project or create a symlink with:

```bash
$ ln -s [path/to/data] data
```

Then run the preprocessing script download the weather data, process, and combine it with the electricity price and installed capacity data:

```bash
(.ba_venv)$ preprocess.py
```

## Running Experiments
The experiment are run with
```bash
(.ba_venv)$ experiment_{N}.py
```
where the numbers `{N}` correspond the the experiments in the thesis paper.
Experiment `experiment_arnn.py` is the test that was used to evaluate the predictive power of the autocorrelation in the price data, as described in the `Results` section of the thesis paper.

The `offset.py` experiment allows to test the effect of the timestep offset as described in the first paragraph of the results section.
`frequencies.py` creates the fourier transformation of the price data as used in the `Data` chapter.
`images.py` can be used to create the visualization of the weather data as shown in the `Data` chapter.
However, this requires to manually download the desired monthly weather data from [here](https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/), place it in the `data/raw_data/weather_data` folder, and play around with the array slice in line 59.
The current file contains the code that was used to create the images as shown in the thesis paper.
