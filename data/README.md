# Data Layout

Place the UT-HAR dataset under the following paths:

```text
data/UT_HAR/data/X_train.csv
data/UT_HAR/data/X_val.csv
data/UT_HAR/data/X_test.csv
data/UT_HAR/label/y_train.csv
data/UT_HAR/label/y_val.csv
data/UT_HAR/label/y_test.csv
```

The files use `.csv` extensions in the original benchmark, but they are NumPy binary files and should be loaded with `np.load`.

For the processed UT-HAR benchmark used in this repository:

- each `X` sample has shape `(250, 90)`
- `250` is interpreted as `CSI frame` indices / `timesteps`
- `90` is interpreted as `CSI features`, which can be explained in `Intel 5300 NIC` style CSI layout as `30 subcarriers x 3 antenna pairs`

Timing conversion requires `sampling_rate` and is not assumed as ground truth in this repository. When time is discussed, it should be phrased as an estimate based on an assumed `sampling_rate`, while `CSI frame count` remains the rigorous quantity.
