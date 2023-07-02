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

    __slots__ = "_data_numeric", "_data_string", "_index", "_columns", "_columns_order", "_data"

    # region magic methods
    def __init__(
        self,
        data_numeric: NDArrayLike[IF],
        data_string: NDArrayLike[np.str_],
        index: NDArrayLike[IFS],
        columns: NDArrayLike[IFS],
        columns_order: NDArrayLike[np.int_],
        data: ch.H5Dict[Any] | None = None,
    ):
        """
        Args:
            file: an optional h5py group where this VDataFrame is read from.
        """
        self._data_numeric = data_numeric
        self._data_string = data_string
        self._index = index
        self._columns = columns
        self._columns_order = columns_order
        self._data = data

    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame, values: ch.H5Dict[Any] | None = None) -> H5DataFrame:
        if isinstance(dataframe, H5DataFrame):
            return dataframe

        _data_numeric = dataframe.select_dtypes(include=[np.number, bool])  # type: ignore[list-item]
        _data_string = dataframe.select_dtypes(exclude=[np.number, bool])  # type: ignore[list-item]
        _index = np.array(dataframe.index)
        _columns = np.array(dataframe.columns)
        _columns_order = npi.indices(np.concatenate((_data_numeric.columns, _data_string.columns)), dataframe.columns)

        if values is not None:
            ch.write_objects(
                values,
                data_numeric=_data_numeric.values.astype(np.float64),
                data_string=_data_string.values.astype(str),
                index=_index,
                columns=_columns,
                columns_order=_columns_order,
            )

        return cls(
            _data_numeric.values.astype(np.float64),
            _data_string.values.astype(str),
            _index,
            _columns,
            _columns_order,
            values,
        )

    def __repr__(self) -> str:
        data = np.hstack((self._data_numeric[:5], self._data_string[:5]))[:, self._columns_order]
        repr_ = repr(pd.DataFrame(data, index=self.index, columns=self.columns))
        repr_ += f"\n[{len(self._index)} rows x {len(self._columns)} columns]"
        return repr_

    def __len__(self) -> int:
        return len(self._index)

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        if values is self._data:
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
            data=values,
        )

    # endregion

    # region predicates
    @property
    def empty(self) -> bool:
        return not len(self._index) and not len(self._columns)

    # endregion

    # region attributes
    @property
    def data(self) -> ch.H5Dict[Any] | None:
        return self._data

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

    # region methods
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.hstack((self._data_numeric, self._data_string), dtype=object)[:, self._columns_order],
            index=self.index,
            columns=self.columns,
        )

    # endregion
