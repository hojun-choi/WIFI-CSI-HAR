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
