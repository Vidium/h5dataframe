from __future__ import annotations

import itertools
from typing import Any, Callable, Hashable, Literal, cast

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._libs import lib
from pandas._typing import QuantileInterpolation
from pandas.core.array_algos.take import take_1d  # type: ignore[import-untyped]
from pandas.core.array_algos.quantile import quantile_compat  # type: ignore[import-untyped]
from pandas.core.construction import extract_array, sanitize_array
from pandas.core.dtypes.cast import np_can_hold_element  # type: ignore[attr-defined]
from pandas.core.dtypes.common import ensure_platform_int  # type: ignore[attr-defined]
from pandas.core.dtypes.missing import isna
from pandas.core.indexers.utils import validate_indices  # type: ignore[import-untyped]
from pandas.core.indexes.base import ensure_index
from pandas.core.internals.array_manager import (  # type: ignore[import-untyped]
    ArrayManager,
    BaseArrayManager,
    DataManager,
    SingleDataManager,
    concat_arrays,
)
from pandas.core.internals.base import ensure_np_dtype, interleaved_dtype  # type: ignore[import-untyped]
from pandas.core.internals.blocks import ensure_block_shape, external_values  # type: ignore[import-untyped]

from h5dataframe._typing import NDArrayLike


def extract_pandas_array(array: NDArrayLike[Any], typ: Literal["ram", "file"]) -> NDArrayLike[Any]:
    if typ == "ram" and not isinstance(array, np.ndarray):
        raise TypeError("Only numpy.ndarrays are allowed in H5DataFrames stored in ram.")

    elif typ == "file" and not isinstance(array, ch.H5Array):
        raise TypeError("Only ch5mpy.H5Arrays are allowed in H5DataFrames loaded from file.")

    return array


def array_equals(left: NDArrayLike[Any], right: NDArrayLike[Any]) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
    if left.dtype != right.dtype and (left.dtype.kind not in "OSU" or right.dtype.kind not in "OSU"):
        return False

    return np.array_equal(left, right)


class H5ArrayManager(BaseArrayManager):  # type: ignore[misc]
    def __init__(
        self,
        arrays: list[NDArrayLike[Any]],
        axes: list[pd.Index[Any]],
        verify_integrity: bool = True,
    ) -> None:
        self.typ: Literal["ram", "file"] = "file" if any(isinstance(arr, ch.H5Array) for arr in arrays) else "ram"

        if verify_integrity:
            self._axes = [ensure_index(ax) for ax in axes]
            self.arrays = [extract_pandas_array(x, self.typ) for x in arrays]
            self._verify_integrity()

        else:
            # Note: we are storing the axes in "_axes" in the (row, columns) order
            # which contrasts the order how it is stored in BlockManager
            self._axes = axes
            self.arrays = arrays

    def _verify_integrity(self) -> None:
        n_columns, n_rows = self.shape

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

            if not arr.ndim == 1:
                raise ValueError(
                    "Passed arrays should be 1-dimensional, got array with " f"{arr.ndim} dimensions instead."
                )

    @property
    def ndim(self) -> Literal[2]:
        return 2

    # --------------------------------------------------------------------
    # Indexing
    def fast_xs(self, loc: int) -> H5SingleArrayManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        values = [arr[loc] for arr in self.arrays]
        result = np.hstack(values, dtype=dtype)
        return H5SingleArrayManager([result], [self._axes[1]])

    def get_slice(self, slobj: slice, axis: int = 0) -> H5ArrayManager:
        axis = self._normalize_axis(axis)

        if axis == 0:
            arrays: list[NDArrayLike[Any]] = [arr[slobj] for arr in self.arrays]

        elif axis == 1:
            arrays = self.arrays[slobj]

        else:
            raise ValueError(f"Invalid axis: {axis}")

        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)  # type: ignore[attr-defined]

        return type(self)(arrays, new_axes, verify_integrity=False)

    def iget(self, i: int) -> H5SingleArrayManager:
        """
        Return the data as a H5SingleArrayManager.
        """
        return H5SingleArrayManager([self.arrays[i]], [self._axes[0]])

    def iget_values(self, i: int) -> NDArrayLike[Any]:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        return self.arrays[i]

    @property
    def column_arrays(self) -> list[npt.NDArray[Any]]:
        """
        Used in the JSON C code to access column arrays.
        """
        return [np.asarray(arr) for arr in self.arrays]

    def iset(
        self,
        loc: int | slice | npt.NDArray[np.bool_],
        value: NDArrayLike[Any],
        inplace: bool = False,
        refs: None = None,
    ) -> None:
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        inplace : bool, default False
            Whether overwrite existing array as opposed to replacing it.
        """
        # single column -> single integer index
        if lib.is_integer(loc):
            # TODO can we avoid needing to unpack this here? That means converting
            # DataFrame into 1D array when loc is an integer
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] == 1
                value = value[:, 0]

            assert isinstance(value, np.ndarray)
            assert value.ndim == 1
            assert len(value) == len(self._axes[0])

            self.arrays[loc] = value
            return

        # multiple columns -> convert slice or array to integer indices
        elif isinstance(loc, slice):
            indices: range | npt.NDArray[Any] = range(
                loc.start if loc.start is not None else 0,
                loc.stop if loc.stop is not None else self.shape_proper[1],
                loc.step if loc.step is not None else 1,
            )

        else:
            assert isinstance(loc, np.ndarray)
            assert loc.dtype == "bool"
            indices = np.nonzero(loc)[0]

        assert value.ndim == 2
        assert value.shape[0] == len(self._axes[0])

        for value_idx, mgr_idx in enumerate(indices):
            value_arr = value[:, value_idx]
            self.arrays[mgr_idx] = value_arr

    def column_setitem(
        self, loc: int, idx: int | slice | npt.NDArray[Any], value: Any, inplace_only: bool = False
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the ArrayManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        if not lib.is_integer(loc):
            raise TypeError("The column index should be an integer")

        arr = self.arrays[loc]
        mgr = H5SingleArrayManager([arr], [self._axes[0]])

        mgr.setitem_inplace(idx, value)

    def insert(self, loc: int, item: Hashable, value: NDArrayLike[Any], refs: None = None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
        value = extract_array(value, extract_numpy=True)
        if value.ndim == 2:
            if value.shape[0] == 1:
                value = value[0, :]

            else:
                raise ValueError(f"Expected a 1D array, got an array with shape {value.shape}")

        self.arrays.insert(loc, value)
        self._axes[1] = self.items.insert(loc, item)

    def idelete(self, indexer: npt.NDArray[np.intp] | None) -> H5ArrayManager:
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False

        self.arrays = [self.arrays[i] for i in np.nonzero(to_keep)[0]]
        self._axes = [self._axes[0], self._axes[1][to_keep]]
        return self

    def _reindex_indexer(
        self,
        new_axis: pd.Index[Any],
        indexer: npt.NDArray[np.intp] | None,
        axis: int,
        fill_value: None = None,
        allow_dups: bool = False,
        copy: bool | None = True,
        use_na_proxy: bool = False,
    ) -> H5ArrayManager:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True


        pandas-indexer with -1's only.
        """
        if copy is None:
            # ArrayManager does not yet support CoW, so deep=None always means
            # deep=True for now
            copy = True

        if indexer is None:
            if new_axis is self._axes[axis] and not copy:
                return self

            new_axes = list(self._axes)
            new_axes[axis] = new_axis
            return H5ArrayManager(self.arrays, new_axes, verify_integrity=False)

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self._axes[axis]._validate_can_reindex(indexer)  # type: ignore[attr-defined]

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 1:
            new_arrays = []
            for i in indexer:
                if i == -1:
                    arr = self._make_na_array(fill_value=fill_value, use_na_proxy=use_na_proxy)
                else:
                    arr = self.arrays[i]

                new_arrays.append(arr)

        else:
            validate_indices(indexer, len(self._axes[0]))
            indexer = ensure_platform_int(indexer)
            mask = cast(npt.NDArray[np.bool_], indexer == -1)
            needs_masking = mask.any()

            new_arrays = [
                take_1d(
                    arr,
                    indexer,
                    allow_fill=needs_masking,
                    fill_value=fill_value,
                    mask=mask,
                )
                for arr in self.arrays
            ]

        new_axes = list(self._axes)
        new_axes[axis] = new_axis

        return H5ArrayManager(new_arrays, new_axes, verify_integrity=False)

    # --------------------------------------------------------------------
    # Array-wise Operation
    def grouped_reduce(self, func: Callable[..., Any]) -> H5ArrayManager:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[npt.NDArray[Any]] = []
        result_indices: list[int] = []

        for i, arr in enumerate(self.arrays):
            # grouped_reduce functions all expect 2D arrays
            arr = ensure_block_shape(arr, ndim=2)
            res = func(arr)
            if res.ndim == 2:
                # reverse of ensure_block_shape
                assert res.shape[0] == 1
                res = res[0]

            result_arrays.append(res)
            result_indices.append(i)

        if len(result_arrays) == 0:
            nrows = 0
        else:
            nrows = result_arrays[0].shape[0]
        index = pd.Index(range(nrows))

        columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        return type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]

    def reduce(self, func: Callable[..., Any]) -> H5ArrayManager:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[npt.NDArray[Any]] = []
        for i, arr in enumerate(self.arrays):
            res = func(arr, axis=0)

            # TODO NaT doesn't preserve dtype, so we need to ensure to create
            # a timedelta result array if original was timedelta
            # what if datetime results in timedelta? (eg std)
            dtype = arr.dtype if res is pd.NaT else None
            result_arrays.append(sanitize_array([res], None, dtype=dtype))

        index = pd.Index._simple_new(np.array([None], dtype=object))  # type: ignore[attr-defined]
        columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        new_mgr = type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]
        return new_mgr

    def operate_blockwise(
        self, other: ArrayManager, array_op: Callable[[NDArrayLike[Any], NDArrayLike[Any]], NDArrayLike[Any]]
    ) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        # TODO what if `other` is BlockManager ?
        left_arrays = self.arrays
        right_arrays = other.arrays
        result_arrays = [array_op(left, right) for left, right in zip(left_arrays, right_arrays)]
        return type(self)(result_arrays, self._axes)

    def quantile(
        self,
        *,
        qs: pd.Index[Any],  # with dtype float64
        transposed: bool = False,
        interpolation: QuantileInterpolation = "linear",
    ) -> ArrayManager:
        arrs = [ensure_block_shape(x, 2) for x in self.arrays]
        new_arrs = [quantile_compat(x, np.asarray(qs._values), interpolation) for x in arrs]  # type: ignore[attr-defined]
        for i, arr in enumerate(new_arrs):
            if arr.ndim == 2:
                assert arr.shape[0] == 1, arr.shape
                new_arrs[i] = arr[0]

        axes = [qs, self._axes[1]]
        return type(self)(new_arrs, axes)

    # ----------------------------------------------------------------
    def unstack(self, unstacker: Any, fill_value: Any) -> H5ArrayManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        indexer, _ = unstacker._indexer_and_to_sort
        if unstacker.mask.all():
            new_indexer = indexer
            allow_fill = False
            new_mask2D = None
            needs_masking = None
        else:
            new_indexer = np.full(unstacker.mask.shape, -1)
            new_indexer[unstacker.mask] = indexer
            allow_fill = True
            # calculating the full mask once and passing it to take_1d is faster
            # than letting take_1d calculate it in each repeated call
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        new_indexer2D = new_indexer.reshape(*unstacker.full_shape)
        new_indexer2D = ensure_platform_int(new_indexer2D)

        new_arrays = []
        for arr in self.arrays:
            for i in range(unstacker.full_shape[1]):
                if allow_fill:
                    # error: Value of type "Optional[Any]" is not indexable  [index]
                    new_arr = take_1d(
                        arr,
                        new_indexer2D[:, i],
                        allow_fill=needs_masking[i],  # type: ignore[index]
                        fill_value=fill_value,
                        mask=new_mask2D[:, i],  # type: ignore[index]
                    )
                else:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=False)
                new_arrays.append(new_arr)

        new_index = unstacker.new_index
        new_columns = unstacker.get_new_columns(self._axes[1])
        new_axes = [new_index, new_columns]

        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def as_array(
        self,
        dtype: Any = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> npt.NDArray[Any]:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : object, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        if len(self.arrays) == 0:
            empty_arr = np.empty(self.shape, dtype=float)
            return empty_arr.transpose()

        # We want to copy when na_value is provided to avoid
        # mutating the original object
        copy = copy or na_value is not lib.no_default

        if not dtype:
            dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        dtype = ensure_np_dtype(dtype)

        result = np.empty(self.shape_proper, dtype=dtype)

        for i, arr in enumerate(self.arrays):
            arr = arr.astype(arr.dtype, copy=copy)
            result[:, i] = arr

        if na_value is not lib.no_default:
            result[isna(result)] = na_value

        return result

    @classmethod
    def concat_horizontal(cls, mgrs: list[H5ArrayManager], axes: list[pd.Index[Any]]) -> H5ArrayManager:
        """
        Concatenate uniformly-indexed ArrayManagers horizontally.
        """
        # concatting along the columns -> combine reindexed arrays in a single manager
        arrays = list(itertools.chain.from_iterable([mgr.arrays for mgr in mgrs]))
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[H5ArrayManager], axes: list[pd.Index[Any]]) -> H5ArrayManager:
        """
        Concatenate uniformly-indexed ArrayManagers vertically.
        """
        # concatting along the rows -> concat the reindexed arrays
        # TODO(ArrayManager) doesn't yet preserve the correct dtype
        arrays = [concat_arrays([mgrs[i].arrays[j] for i in range(len(mgrs))]) for j in range(len(mgrs[0].arrays))]
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

    def copy(self, deep: bool = True) -> BaseArrayManager:
        if not deep:
            return self

        return ArrayManager([arr.copy() for arr in self.arrays], self._axes, verify_integrity=False)


class H5SingleArrayManager(BaseArrayManager, SingleDataManager):  # type: ignore[misc]
    __slots__ = [
        "_axes",  # private attribute, because 'axes' has different order, see below
        "arrays",
    ]

    arrays: list[NDArrayLike[Any]]
    _axes: list[pd.Index[Any]]

    def __init__(
        self,
        arrays: list[NDArrayLike[Any]],
        axes: list[pd.Index[Any]],
        verify_integrity: bool = True,
    ) -> None:
        if verify_integrity:
            assert len(axes) == 1
            assert len(arrays) == 1

            typ: Literal["ram", "file"] = "file" if isinstance(arrays[0], ch.H5Array) else "ram"

            self._axes = [ensure_index(ax) for ax in axes]
            self.arrays = [extract_pandas_array(arrays[0], typ)]
            self._verify_integrity()

        else:
            self._axes = axes
            self.arrays = arrays

    def _verify_integrity(self) -> None:
        assert len(self.arrays) == 1

        (n_rows,) = self.shape
        arr = self.arrays[0]
        assert len(arr) == n_rows

        if not arr.ndim == 1:
            raise ValueError("Passed array should be 1-dimensional, got array with " f"{arr.ndim} dimensions instead.")

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @staticmethod
    def _normalize_axis(axis: Any) -> Any:
        return axis

    def make_empty(self, axes: list[pd.Index[Any]] | None = None) -> H5SingleArrayManager:
        """Return an empty ArrayManager with index/array of length 0"""
        if axes is None:
            axes = [pd.Index([], dtype=object)]

        array: npt.NDArray[Any] = np.array([], dtype=self.dtype)
        return type(self)([array], axes)

    @classmethod
    def from_array(cls, array: npt.NDArray[Any], index: pd.Index[Any]) -> H5SingleArrayManager:
        return cls([array], [index])

    @property
    def axes(self) -> list[pd.Index[Any]]:
        return self._axes

    @property
    def index(self) -> pd.Index[Any]:
        return self._axes[0]

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(np.dtype[Any], self.array.dtype)

    def external_values(self) -> NDArrayLike[Any]:
        """The array that Series.values returns"""
        return cast(NDArrayLike[Any], external_values(self.array))

    def internal_values(self) -> NDArrayLike[Any]:
        """The array that Series._values returns"""
        return cast(NDArrayLike[Any], self.array)

    # def array_values(self):
    #     """The array that Series.array returns"""
    #     arr = self.array
    #     if isinstance(arr, np.ndarray):
    #         arr = NumpyExtensionArray(arr)
    #     return arr

    @property
    def _can_hold_na(self) -> bool:
        if isinstance(self.array, np.ndarray):
            return self.array.dtype.kind not in "iub"
        else:
            # ExtensionArray
            raise NotImplementedError
            # return self.array._can_hold_na

    @property
    def is_single_block(self) -> bool:
        return True

    def equals(self, other: DataManager) -> bool:
        if not isinstance(other, DataManager):
            return False

        self_axes, other_axes = self.axes, other.axes

        if len(self_axes) != len(other_axes):
            return False

        if not all(ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes)):
            return False

        for left, right in zip(self.arrays, other.arrays):
            if not array_equals(left, right):
                return False
        return True

    def fast_xs(self, loc: int) -> H5SingleArrayManager:
        raise NotImplementedError("Use series._values[loc] instead")

    def get_slice(self, slobj: slice, axis: int = 0) -> H5SingleArrayManager:
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)  # type: ignore[attr-defined]
        return type(self)([new_array], [new_index], verify_integrity=False)

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> H5SingleArrayManager:
        new_array = self.array[indexer]
        new_index = self.index[indexer]
        return type(self)([new_array], [new_index])

    def apply(self, func: Callable[[NDArrayLike[Any]], NDArrayLike[Any]], **kwargs: Any) -> H5SingleArrayManager:
        if callable(func):
            new_array = func(self.array, **kwargs)
        else:
            new_array = getattr(self.array, func)(**kwargs)
        return type(self)([new_array], self._axes)

    def setitem(self, indexer: Any, value: Any, warn: bool = True) -> H5SingleArrayManager:
        """
        Set values with indexer.

        For SingleArrayManager, this backs s[indexer] = value

        See `setitem_inplace` for a version that works inplace and doesn't
        return a new Manager.
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")

        return cast(H5SingleArrayManager, self.apply_with_block("setitem", indexer=indexer, value=value))

    def setitem_inplace(self, indexer: Any, value: Any, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        arr = self.array

        # EAs will do this validation in their own __setitem__ methods.
        if isinstance(arr, np.ndarray) and not isinstance(arr, ch.H5Array):
            # Note: checking for ndarray instead of np.dtype means we exclude
            #  dt64/td64, which do their own validation.
            value = np_can_hold_element(arr.dtype, value)

        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == 1:
            # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
            value = value[0, ...]

        arr[indexer] = value

    # def idelete(self, indexer) -> H5SingleArrayManager:
    #     """
    #     Delete selected locations in-place (new array, same ArrayManager)
    #     """
    #     to_keep = np.ones(self.shape[0], dtype=np.bool_)
    #     to_keep[indexer] = False
    #
    #     self.arrays = [self.arrays[0][to_keep]]
    #     self._axes = [self._axes[0][to_keep]]
    #     return self

    def _get_data_subset(self, predicate: Callable[[NDArrayLike[Any]], bool]) -> H5SingleArrayManager:
        # used in get_numeric_data / get_bool_data
        if predicate(self.array):
            return type(self)(self.arrays, self._axes, verify_integrity=False)

        else:
            return self.make_empty()

    def set_values(self, values: NDArrayLike[Any]) -> None:
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
        self.arrays[0] = values

    def to_2d_mgr(self, columns: pd.Index[Any]) -> H5ArrayManager:
        """
        Manager analogue of Series.to_frame
        """
        arrays = [self.arrays[0]]
        axes = [self.axes[0], columns]

        return H5ArrayManager(arrays, axes, verify_integrity=False)
