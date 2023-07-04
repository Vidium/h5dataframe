from __future__ import annotations

import re
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
        data: pd.DataFrame | dict[Hashable, NDArrayLike[Any] | ExtensionArray] | ch.H5Dict[Any] | None = None,
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

        elif isinstance(data, pd.DataFrame):
            _index = data.index if index is None else pd.Index(index)
            _columns = data.columns if columns is None else pd.Index(columns)
            arrays = [np.array(arr) for arr in data.to_dict(orient="list").values()]
            file = None

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

        else:
            raise TypeError(f"Invalid type '{type(data)}' for 'data' argument.")

        mgr = ArrayManager(arrays, [_index, _columns])
        object.__setattr__(self, "_data_file", file)

        NDFrame.__init__(self, mgr)  # type: ignore[call-arg]

    def __repr__(self) -> str:
        repr_ = repr(self.iloc[:5].copy())
        if self.empty:
            return repr_

        re.sub(r"\n\n\[.*\]$", "", repr_)
        return (
            repr_ + f"\n{'[RAM]' if self._data_file is None else '[FILE]'}\n"
            f"[{len(self.index)} rows x {len(self.columns)} columns]"
        )

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

    def write(self, file: str | Path | ch.File | ch.Group | ch.H5Dict[Any], name: str = "") -> None:
        if not isinstance(file, (ch.H5Dict, ch.Group)):
            file = ch.File(file, mode=ch.H5Mode.WRITE_TRUNCATE)

        if not isinstance(file, ch.H5Dict):
            file = ch.H5Dict(file)

        ch.write_object(file, name, self)

        if self._data_file is None:
            mgr = ArrayManager(
                [arr for arr in file["arrays"].values()], [pd.Index(file["index"]), pd.Index(file["arrays"].keys())]
            )
            self._data_file = file

            NDFrame.__init__(self, mgr)  # type: ignore[call-arg]

    # endregion
