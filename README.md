# h5dataframe

Drop-in replacement for pandas DataFrames that allows to store data on an hdf5 file and manipulate data directly from that h5df file without loading it in memory.

# Warning !

This is very much a **work in progress**, some features might not work yet or cause bugs.
**Save** your data elsewhere before converting it to an H5DataFrame.

If you miss a feature from pandas DataFrames, please fill an issue or feel free to contribute.

# Overview

This library provides the `H5DataFrame` object, replacing the regular `pandas.DataFrame`.

An `H5DataFrame` can be created from a `pandas.DataFrame` or from a dictionnary of (column_name -> column_values).

```python
>>> import pandas as pd
>>> from h5dataframe import H5DataFrame
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, 
                      index=['r1', 'r2', 'r3'])
>>> hdf = H5DataFrame(df)
>>> hdf
    a  b
r1  1  4
r2  2  5
r3  3  6
[RAM]
[3 rows x 2 columns]
```

At this point, all the data is still loaded in RAM, as indicated by the second-to-last line. To write the data to an h5df file, use the `H5DataFrame.write()` method.

```python
>>> hdf.write('path/to/file.h5')
>>> hdf
    a  b
r1  1  4
r2  2  5
r3  3  6
[FILE]
[3 rows x 2 columns]
```

The `H5DataFrame` is now backed on an h5df file, only loading data in RAM when requested.

Alternatively, an `H5DataFrame` can be read directly from an previously created h5df file with the `H5DataFrame.read()` method.

```python
>>> from h5dataframe import H5Mode
>>> H5DataFrame.read('path/to/file.h5', mode=H5Mode.READ)
    a  b
r1  1  4
r2  2  5
r3  3  6
[FILE]
[3 rows x 2 columns]
```

The default mode is `READ` (`'r'`) which creates a **read-only** `H5DataFrame`. To modify the data, use `mode=H5Mode.READ_WRITE` (`'r+'`).

# Installation

From pip:
```shell
pip install h5dataframe
```

From source:
```shell
git clone git@github.com:Vidium/h5dataframe.git
```