from typing import Any, TypeVar

import ch5mpy as ch
import numpy as np
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries  # type: ignore[attr-defined]
from pandas.core.internals.array_manager import ArrayManager as pd_ArrayManager  # type: ignore[import-untyped]
from pandas.core.internals.array_manager import BaseArrayManager as pd_BaseArrayManager
from pandas.core.internals.blocks import ensure_block_shape  # type: ignore[import-untyped]

from h5dataframe.block import new_block

T = TypeVar("T", bound=pd_BaseArrayManager)


# patch for support of H5Arrays
def apply_with_block(self: T, f: str, align_keys: list[str] | None = None, swap_axis: bool = True, **kwargs: Any) -> T:
    # switch axis to follow BlockManager logic
    if swap_axis and "axis" in kwargs and self.ndim == 2:
        kwargs["axis"] = 1 if kwargs["axis"] == 0 else 0

    align_keys = align_keys or []
    aligned_args = {k: kwargs[k] for k in align_keys}

    result_arrays = []

    for i, arr in enumerate(self.arrays):
        if aligned_args:
            for k, obj in aligned_args.items():
                if isinstance(obj, (ABCSeries, ABCDataFrame)):  # type: ignore[misc, arg-type]
                    # The caller is responsible for ensuring that
                    #  obj.axes[-1].equals(self.items)
                    if obj.ndim == 1:
                        if self.ndim == 2:
                            kwargs[k] = obj.iloc[slice(i, i + 1)]._values
                        else:
                            kwargs[k] = obj.iloc[:]._values
                    else:
                        kwargs[k] = obj.iloc[:, [i]]._values
                else:
                    # otherwise we have an ndarray
                    if obj.ndim == 2:
                        kwargs[k] = obj[[i]]

        if isinstance(arr.dtype, np.dtype) and not isinstance(arr, (np.ndarray, ch.H5Array)):
            # i.e. TimedeltaArray, DatetimeArray with tz=None. Need to
            #  convert for the Block constructors.
            arr = np.asarray(arr)

        if self.ndim == 2:
            arr = ensure_block_shape(arr, 2)
            block = new_block(arr, placement=slice(0, 1, 1), ndim=2)
        else:
            block = new_block(arr, placement=slice(0, len(self), 1), ndim=1)

        applied = getattr(block, f)(**kwargs)
        if isinstance(applied, list):
            applied = applied[0]
        arr = applied.values
        if self.ndim == 2 and arr.ndim == 2:
            # 2D for np.ndarray or DatetimeArray/TimedeltaArray
            assert len(arr) == 1
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[int, slice]"
            arr = arr[0, :]
        result_arrays.append(arr)

    return type(self)(result_arrays, self._axes)  # type: ignore[no-any-return]


pd_BaseArrayManager.apply_with_block = apply_with_block


class ArrayManager(pd_ArrayManager):  # type: ignore[misc]
    # region methods
    def _verify_integrity(self) -> None:
        n_rows, n_columns = self.shape_proper
        if not len(self.arrays) == n_columns:
            raise ValueError(
                "Number of passed arrays must equal the size of the column Index: "
                f"{len(self.arrays)} arrays vs {n_columns} columns."
            )
        for arr in self.arrays:
            if not len(arr) == n_rows:
                raise ValueError(
                    "Passed arrays should have the same length as the rows Index: " f"{len(arr)} vs {n_rows} rows"
                )
            if not isinstance(arr, (np.ndarray, ExtensionArray, ch.H5Array)):
                raise ValueError(
                    "Passed arrays should be np.ndarray or ExtensionArray instances, " f"got {type(arr)} instead"
                )
            if not arr.ndim == 1:
                raise ValueError(
                    "Passed arrays should be 1-dimensional, got array with " f"{arr.ndim} dimensions instead."
                )

    # endregion
