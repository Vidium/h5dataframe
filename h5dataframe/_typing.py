from typing import TypeVar, Union

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

_T = TypeVar("_T", bound=np.generic, covariant=True)

IFS = Union[np.int_, np.float_, np.str_]
NDArrayLike = Union[npt.NDArray[_T], ch.H5Array[_T]]
