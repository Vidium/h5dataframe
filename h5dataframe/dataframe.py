from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Hashable, Literal

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.core.generic import NDFrame
from pandas.core.internals.array_manager import BaseArrayManager  # type: ignore[import-untyped]

from h5dataframe._typing import IFS, NDArrayLike
from h5dataframe.manager import H5ArrayManager


class H5DataFrame(pd.DataFrame):
    """
    Proxy for pandas DataFrames for storing data in an hdf5 file.
    """

    _internal_names = ["_data_file"] + pd.DataFrame._internal_names  # type: ignore[attr-defined]
    _internal_names_set = {"_data_file"} | pd.DataFrame._internal_names_set  # type: ignore[attr-defined]

    _data_file: ch.H5Dict[Any] | None

    # region magic methods
    @property
    def _constructor(self) -> Callable[..., H5DataFrame]:
        def inner(df: H5DataFrame) -> H5DataFrame:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("H5DataFrame constructor not properly called")

            return df

        return inner

    def _constructor_from_mgr(self, mgr: BaseArrayManager, axes: list[pd.Index[Any]]) -> pd.DataFrame:
        if not isinstance(mgr, H5ArrayManager):
            df: pd.DataFrame = pd.DataFrame._from_mgr(mgr, axes=axes)  # type: ignore[attr-defined]
        else:
            df = H5DataFrame._from_mgr(mgr, axes=axes)  # type: ignore[attr-defined]

        if isinstance(self, pd.DataFrame):
            # This would also work `if self._constructor is DataFrame`, but
            #  this check is slightly faster, benefiting the most-common case.
            return df

        elif type(self).__name__ == "GeoDataFrame":
            # Shim until geopandas can override their _constructor_from_mgr
            #  bc they have different behavior for Managers than for DataFrames
            return self._constructor(mgr)

        # We assume that the subclass __init__ knows how to handle a
        #  pd.DataFrame object.
        return self._constructor(df)

    def _constructor_sliced_from_mgr(self, mgr: BaseArrayManager, axes: list[pd.Index[Any]]) -> pd.Series[Any]:
        ser: pd.Series[Any] = pd.Series._from_mgr(mgr, axes)  # type: ignore[attr-defined]
        ser._name = None  # caller is responsible for setting real name

        if isinstance(self, pd.DataFrame):
            return ser

        return self._constructor_sliced(ser)

    def __init__(
        self,
        data: pd.DataFrame | dict[Hashable, NDArrayLike[Any]] | ch.H5Dict[Any] | npt.NDArray[Any] | None = None,
        index: NDArrayLike[IFS] | pd.Index[Any] | None = None,
        columns: NDArrayLike[IFS] | pd.Index[Any] | None = None,
    ):
        if isinstance(data, ch.H5Dict):
            assert index is None
            assert columns is None

            _index: pd.Index[Any] = pd.Index(data["index"].copy())
            _columns: pd.Index[Any] = pd.Index(data["arrays"].keys(), dtype=data.attributes["columns_dtype"])
            arrays: list[NDArrayLike[Any]] = [arr for arr in data["arrays"].values()]
            file = data

        elif isinstance(data, pd.DataFrame):
            _index = data.index if index is None else pd.Index(index)
            _columns = data.columns if columns is None else pd.Index(columns)

            # FIXME: SOMETIMES only does not work once then works, can't debug WTF
            #  ValueError: Does not understand character buffer dtype format string ('w')
            try:
                arrays = [np.array(arr) for arr in data.to_dict(orient="list").values()]
            except ValueError:
                arrays = [np.array(arr) for arr in data.to_dict(orient="list").values()]

            file = None

        elif isinstance(data, np.ndarray):
            _index = pd.RangeIndex(start=0, stop=data.shape[0]) if index is None else pd.Index(index)
            _columns = pd.RangeIndex(start=0, stop=data.shape[1]) if columns is None else pd.Index(columns)
            arrays = [col for col in data.T]
            file = None

        elif isinstance(data, dict):
            assert columns is None
            _columns = pd.Index(data.keys())
            arrays = list(data.values())
            _index = pd.RangeIndex(start=0, stop=arrays[0].shape[0]) if index is None else pd.Index(index)
            file = None

        elif data is None:
            _index = pd.RangeIndex(0) if index is None else pd.Index(index)
            _columns = pd.RangeIndex(0) if columns is None else pd.Index(columns)
            arrays = []
            file = None

        else:
            raise TypeError(f"Invalid type '{type(data)}' for 'data' argument.")

        mgr = H5ArrayManager(arrays, [_index, _columns])
        object.__setattr__(self, "_data_file", file)

        NDFrame.__init__(self, mgr)  # type: ignore[call-arg]

    def __finalize__(self, other: H5DataFrame, method: str | None = None, **kwargs: Any) -> pd.DataFrame:  # type: ignore[override]
        super().__finalize__(other, method, **kwargs)

        if method == "copy":
            return other

        return self

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

        values.attributes["columns_dtype"] = self.columns.dtype
        # TODO: fix typing in ch5mpy
        values["index"] = self.index.values
        values["arrays"] = {str(k): v for k, v in self.to_dict(orient="list").items()}

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

        ch.write_object(self, file, name)

        if self._data_file is None:
            mgr = H5ArrayManager(
                [arr for arr in file["arrays"].values()],
                [pd.Index(file["index"].copy()), pd.Index(file["arrays"].keys())],
            )
            self._data_file = file

            NDFrame.__init__(self, mgr)  # type: ignore[call-arg]

    # endregion
