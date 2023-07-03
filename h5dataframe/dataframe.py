from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Literal

import ch5mpy as ch
import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray
from pandas.core.generic import NDFrame

from h5dataframe._typing import IFS, NDArrayLike
from h5dataframe.manager import ArrayManager


class H5DataFrame(pd.DataFrame):
    """
    Proxy for pandas DataFrames for storing data in an hdf5 file.
    """

    _internal_names = ["_data_file"] + pd.DataFrame._internal_names  # type: ignore[attr-defined]
    _internal_names_set = {"_data_file"} | pd.DataFrame._internal_names_set  # type: ignore[attr-defined]

    _data_file: ch.H5Dict[Any] | None

    # region magic methods
    def __init__(
        self,
        data: dict[Hashable, NDArrayLike[Any] | ExtensionArray] | ch.H5Dict[Any] | None = None,
        index: NDArrayLike[IFS] | None = None,
        columns: NDArrayLike[IFS] | None = None,
    ):
        if isinstance(data, ch.H5Dict):
            assert index is None
            assert columns is None

            _index: pd.Index | None = pd.Index(data["index"])
            _columns: pd.Index | None = pd.Index(data["arrays"].keys())
            arrays = [arr for arr in data["arrays"].values()]
            file = data

        elif isinstance(data, dict):
            assert columns is None
            _columns = pd.Index(data.keys())
            arrays = list(data.values())
            _index = pd.RangeIndex(start=0, stop=arrays[0].shape[0]) if index is None else pd.Index(index)
            file = None

        elif data is None:
            _index = None if index is None else pd.Index(index)
            _columns = None if columns is None else pd.Index(columns)
            arrays = []
            file = None

        mgr = ArrayManager(arrays, [_index, _columns])
        object.__setattr__(self, "_data_file", file)

        NDFrame.__init__(self, mgr)  # type: ignore[call-arg]

    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame, values: ch.H5Dict[Any] | None = None) -> H5DataFrame:
        if isinstance(dataframe, H5DataFrame):
            return dataframe

        _index = np.array(dataframe.index)
        _arrays = {col.name: col.values for _, col in dataframe.items()}

        if values is not None:
            values["index"] = _index
            values["arrays"] = _arrays

            return cls(values)

        return cls(_arrays, index=_index)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return repr(self.iloc[:5].copy()) + f"\n[{len(self.index)} rows x {len(self.columns)} columns]"

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        if values is self._data:
            return

        values["index"] = self.index
        values["arrays"] = self.to_dict(orient="list")

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict[Any]) -> H5DataFrame:
        return H5DataFrame(values)

    # endregion

    # region attributes
    @property
    def data(self) -> ch.H5Dict[Any] | None:
        return self._data_file

    # endregion

    # region methods
    @classmethod
    def read(
        cls, path: str | Path | ch.H5Dict[Any], mode: Literal[ch.H5Mode.READ, ch.H5Mode.READ_WRITE] = ch.H5Mode.READ
    ) -> H5DataFrame:
        if not isinstance(path, ch.H5Dict):
            path = ch.H5Dict.read(path, mode=mode)

        return cls.__h5_read__(path)

    # endregion
