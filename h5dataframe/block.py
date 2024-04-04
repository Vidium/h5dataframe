from __future__ import annotations

from typing import Any

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
from pandas._libs.internals import BlockValuesRefs  # type: ignore[import-untyped]
from pandas.core.internals.blocks import Block  # type: ignore[import-untyped]
from pandas.core.internals.blocks import new_block as pd_new_block


class H5Block:
    # region magic methods
    def __init__(self, values: ch.H5Array[Any], ndim: int, placement: slice, refs: None) -> None:
        assert isinstance(placement, slice)
        assert refs is None
        assert ndim == 1

        self.values = values[placement]

    # endregion

    # region methods
    def setitem(self, indexer: tuple[int], value: Any) -> H5Block:
        self.values[indexer] = value
        return self

    # endregion


def new_block(
    values: npt.NDArray[Any] | ch.H5Array[Any], placement: slice, *, ndim: int, refs: BlockValuesRefs | None = None
) -> Block | H5Block:
    if isinstance(values, np.ndarray):
        return pd_new_block(values, placement, ndim=ndim, refs=refs)

    return H5Block(values, ndim, placement, refs)
