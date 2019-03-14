# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities
=========

**Module name:** :mod:`pennylane.utils`

.. currentmodule:: pennylane.utils

This module contains utilities and auxiliary functions, which are shared
across the PennyLane submodules.

.. raw:: html

    <h3>Summary</h3>

.. autosummary::
    _flatten
    _unflatten
    unflatten

.. raw:: html

    <h3>Code details</h3>
"""
from collections.abc import Iterable
import numbers

import autograd.numpy as np

from .variable  import Variable


def _flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, (numbers.Number, Variable, str)):
        return flat[0], flat[1:]
    elif isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]
    elif isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError('Flattened iterable has more elements than the model.')
    return res


def strongly_entangling_circuit_weights(n_layers, n_wires, mean=0., std=1., uniform=False):
    """
    Creates a numpy array of weights for a Strongly Entangling Circuit.
    The array has the correct shape to use in the function :ref:`REF` and draws each weight
    from a normal distribution or, if `uniform` is `True`, uniformly from the interval [0, 2*pi].

    Args:
        n_layers (int): Number of layers
        n_wires (int): Number of qubits
        mean (float): Mean of normal distribution
        std (float): Standard deviation of normal distribution from which parameters are drawn
        uniform (bool): If true, draw weights from [0, 2*pi]

    Returns: Numpy array of weights with shape (n_layers, n_wires, 3)

    """
    if uniform:
        return np.random.randn(n_layers, n_wires, 3)
    else:
        return np.random.normal(loc=mean, scale=std, size=(n_layers, n_wires, 3))


def strongly_entangling_layer_weights(n_wires, mean=0., std=1., uniform=False):
    """
    Creates a numpy array of weights for a layer of the Strongly Entangling Circuit.
    The array has the correct shape to use in the function :ref:`REF` and draws each weight
    from a normal distribution or, if `uniform` is `True`, uniformly from the interval [0, 2*pi].

    Args:

        n_wires (int): Number of qubits
        mean (float): Mean of normal distribution
        std (float): Standard deviation of normal distribution from which parameters are drawn
        uniform (bool): If true, draw weights from [0, 2*pi]

    Returns: Numpy array of weights with shape (n_wires, 3).

    """
    if uniform:
        return np.random.randn(n_wires, 3)
    else:
        return np.random.normal(loc=mean, scale=std, size=(n_wires, 3))


def cvqnn_circuit_weights(n_layers, n_wires, mean=0., std=0.1, uniform=[0, 2*np.pi]):
    """
    Creates a numpy array of weights for a Continuous-Variable Quantum Neural Net Circuit.
    The array has the correct shape to use in the function :ref:`REF` and draws the weight to
    squeezing, displacement and kerr gates from a normal distribution (specified by `mean` and `std`),
    and the gates of beam splitters uniformly at random from the interval specified in `uniform`.

    Args:
        n_layers (int): Number of layers
        n_wires (int): Number of qubits
        mean (float): Mean of normal distribution
        std (float): Standard deviation of normal distribution from which parameters are drawn
        uniform (array): Draw weights of interferometer from this interval

    Returns: Weights array of shape (n_layers, 5, n_wires), Interferometer weights array of
             shape (n_layers, 6, n_wires*(n_wires-1)//2)

    """

    #TODO: INitialise phases in weights with interval
    n_if = n_wires * (n_wires - 1) // 2
    weights = np.random.normal(loc=mean, scale=std, size=(n_layers, 5, n_wires))
    weights_if = 2 * np.pi * np.random.rand(n_layers, 6, n_if)

    return weights, weights_if


def cvqnn_layer_weights(n_wires, mean=0., std=1., uniform=False):
    """
    Creates a numpy array of weights for a layer of the Continuous-Variable Quantum Neural Net Circuit.
    The array has the correct shape to use in the function :ref:`REF`.

    Args:

        n_wires (int): Number of qubits
        mean (float): Mean of normal distribution
        std (float): Standard deviation of normal distribution from which parameters are drawn
        uniform (array): Draw weights of interferometer from this interval


    Returns: Weights array of shape (5, n_wires), Interferometer weights array of shape (6, n_wires*(n_wires-1)//2)

    """

    #TODO: INitialise phases in weights with interval

    n_if = n_wires*(n_wires-1)//2
    weights = np.random.normal(loc=mean, scale=std, size=(5, n_wires))
    weights_if = 2 * np.pi * np.random.rand(6, n_if)
    return weights, weights_if
