from typing import Any

import xarray

def xarray_acc(
    xs: xarray.DataArray | None, 
    x: xarray.DataArray, 
    dim: Any,
    maxlen: int | None = None,
):
    if xs is None:
        return x
    res = xarray.concat(
        [xs, x], dim=dim,
    )
    if maxlen is not None:
        if res.sizes[dim] > maxlen:
            return res.isel({
                dim: slice(-maxlen, None)
            })
    return res

