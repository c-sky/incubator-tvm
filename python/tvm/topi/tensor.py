# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
from . import cpp
from .. import te, tir
from .utils import get_const_tuple, prod


def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.te.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full(shape, dtype, fill_value)


def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full_like(x, fill_value)


def hardmax(data, axis):
    """hardmax operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape same as input data.
    """

    def _hardmax(data, out_buf, axis):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        out = ib.buffer_ptr(out_buf)
        temp = ib.allocate("float32", (1,), name="temp_data", scope="local")
        temp1 = ib.allocate("int32", (1,), name="temp1_data", scope="local")
        shape = get_const_tuple(data.shape)
        if axis.value == -1:
            axis = len(data.shape) - 1
        else:
            axis = axis.value
        out_size = prod(data.shape)

        with ib.for_range(0, out_size) as i:
            out[i] = 0.0

        outer_size = 1
        for i, s in enumerate(shape):
            if i < axis:
                outer_size = outer_size * s

        inner_size = 1
        for i, s in enumerate(shape):
            if i > axis:
                inner_size = inner_size * s

        cnt = shape[axis]
        with ib.for_range(0, outer_size, "i0") as i:
            with ib.for_range(0, inner_size, "k") as k:
                temp[0] = -float("inf")
                temp1[0] = i * inner_size * cnt + k
                with ib.for_range(0, cnt, "j") as j:
                    index = i * inner_size * cnt + j * inner_size + k
                    with ib.if_scope(input_data[index] > temp[0]):
                        temp1[0] = index
                        temp[0] = input_data[index]

                out[temp1[0]] = 1.0

        return ib.get()

    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: _hardmax(ins[0], outs[0], axis),
        dtype=data.dtype,
    )

    return out
