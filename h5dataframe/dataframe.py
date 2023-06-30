from __future__ import annotations

from typing import Any, Collection

import ch5mpy as ch
import numpy as np
import numpy_indexed as npi
import pandas as pd

from h5dataframe._typing import IF, IFS, NDArrayLike


class H5DataFrame:
    """
    Proxy for pandas DataFrames for storing data in an hdf5 file.
    """

    __slots__ = "_data_numeric", "_data_string", "_index", "_columns", "_columns_order", "_file"

    # region magic methods
    def __init__(
        self,
        data_numeric: NDArrayLike[IF],
        data_string: NDArrayLike[np.str_],
        index: NDArrayLike[IFS],
        columns: NDArrayLike[IFS],
        columns_order: NDArrayLike[np.int_],
        file: ch.H5Dict[Any] | None = None,
    ):
        """
        Args:
            file: an optional h5py group where this VDataFrame is read from.
        """
        assert data_numeric.ndim == 2 and data_string.ndim == 2
        assert data_numeric.shape[0] == data_string.shape[0]

        self._data_numeric = data_numeric.astype(np.float64)
        self._data_string = data_string.astype(str)
        self._index = index
        self._columns = columns
        self._columns_order = columns_order
        self._file = file

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> H5DataFrame:
        _data_numeric = dataframe.select_dtypes(include=[np.number, bool])  # type: ignore[list-item]
        _data_string = dataframe.select_dtypes(exclude=[np.number, bool])  # type: ignore[list-item]

        return cls(
            _data_numeric.values,
            _data_string.values,
            np.array(dataframe.index),
            np.array(dataframe.columns),
            npi.indices(np.concatenate((_data_numeric.columns, _data_string.columns)), dataframe.columns),
        )

    def __repr__(self) -> str:
        data = np.hstack((self._data_numeric[:5], self._data_string[:5]))[:, self._columns_order]
        repr_ = repr(pd.DataFrame(data, index=self.index, columns=self.columns))
        repr_ += f"\n[{len(self._index)} rows x {len(self._columns)} columns]"
        return repr_

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        if self._file is not None:
            return

        ch.write_objects(
            values,
            data_numeric=self._data_numeric,
            data_string=self._data_string,
            index=self._index,
            columns=self._columns,
            columns_order=self._columns,
        )

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> H5DataFrame:
        return H5DataFrame(
            values["data_numeric"],
            values["data_string"],
            values["index"],
            values["columns"],
            values["columns_order"],
            file=values,
        )

    # endregion

    # region attributes
    @property
    def is_backed(self) -> bool:
        """
        Is this VDataFrame backed on a h5 file ?

        Returns:
            Is this VDataFrame backed on a h5 file ?
        """
        return self._file is not None

    @property
    def file(self) -> ch.H5Dict[Any] | None:
        """
        Get the h5 file this VDataFrame is backed on.
        :return: the h5 file this VDataFrame is backed on.
        """
        return self._file

    @property
    def index(self) -> pd.Index:
        """
        Get the index.
        """
        return pd.Index(self._index)

    @index.setter
    def index(self, values: Collection[IFS]) -> None:
        self._index[()] = values

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns.
        """
        return pd.Index(self._columns)

    @columns.setter
    def columns(self, values: Collection[IFS]) -> None:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int]:
        return len(self._index), len(self._columns)

    # endregion
