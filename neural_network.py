from __future__ import annotations
import numpy as np
from typing import Any, List
from copy import deepcopy

def softmax(x : np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def relu(x : np.ndarray | float, a : float = 1) -> np.ndarray | float:
    return (x > 0) * a * x

class NeuralNetwork:
    def __init__(self) -> None:
        self.__input_layer_added  = False
        self.__output_layer_added = False

        self.__shape : List[int] = []
        self.weights : List[np.ndarray] = []
        self.activation_functions : List[np.ufunc] = []

    @property
    def shape(self) -> List[int]:
        return self.__shape

    def copy(self) -> NeuralNetwork:
        return deepcopy(self)
    
    def add_input_layer(self, input_layer_size : int) -> None:
        assert not self.__input_layer_added and not self.__output_layer_added

        self.__shape += [input_layer_size]

        self.__input_layer_added = True

    def add_hidden_layer(self, hidden_layer_size : int, activation_function : np.ufunc) -> None:
        assert self.__input_layer_added and not self.__output_layer_added

        self.__shape += [hidden_layer_size]
        self.activation_functions += [activation_function]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]

    def add_output_layer(self, output_layer_size : int, activation_function) -> None:
        assert self.__input_layer_added and not self.__output_layer_added
        
        self.__shape += [output_layer_size]
        self.activation_functions += [activation_function]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]
        
        self.__output_layer_added = True
    
    def __call__(self, input_layer : np.ndarray) -> np.ndarray:
        assert self.__input_layer_added and self.__output_layer_added
        assert len(input_layer) == self.__shape[0]

        y = np.array(input_layer)
        for w, af in zip(self.weights, self.activation_functions):
            y = af(w @ y)

        return y
    
    def load_from_nn(self, path : str):
        ws = read_nn(path)
        for i, w in enumerate(ws):
            self.weights[i] = w


def float_type(width: int):
    if width == 2:
        return np.float16
    if width == 4:
        return np.float32
    if width == 8:
        return np.float64
    assert False


def read_nn(path: str):
    with open(path, 'rb') as file:
        binary_data = file.read()
    
    
    pa_ti_size_t = binary_data[:4]
    binary_data = binary_data[4:]
    
    pa_ti_T = binary_data[:4]
    binary_data = binary_data[4:]
        
    sizeof_size_t = pa_ti_size_t[0]
    size_t_endianness = 'big' if pa_ti_size_t[1] else 'little'
    size_t_is_integral = not bool(pa_ti_size_t[2])
    size_t_is_signed = bool(pa_ti_size_t[3])
    assert size_t_is_integral


    sizeof_T = pa_ti_T[0]
    T_endianness = 'big' if pa_ti_T[1] else 'little'
    T_is_float = bool(pa_ti_T[2])
    assert T_is_float


    elements_n = binary_data[:sizeof_size_t]
    elements_n = int.from_bytes(elements_n, byteorder=size_t_endianness, signed=size_t_is_signed)
    binary_data = binary_data[sizeof_size_t:]

    matrices_bytes_sizes = []
    for _ in range(elements_n):
        rows = binary_data[:sizeof_size_t]
        binary_data = binary_data[sizeof_size_t:]

        cols = binary_data[:sizeof_size_t]
        binary_data = binary_data[sizeof_size_t:]

        rows = int.from_bytes(rows, byteorder=size_t_endianness, signed=size_t_is_signed)
        cols = int.from_bytes(cols, byteorder=size_t_endianness, signed=size_t_is_signed)

        matrices_bytes_sizes.append(rows * cols * sizeof_T + 4 + 4 + 2 * sizeof_size_t)
    
    result = []
    for bs in matrices_bytes_sizes:
        
        mv = binary_data[:bs]
        binary_data = binary_data[bs:]

        result.append(mv_to_np(mv))
    return result

def mv_to_np(matrix_bytes: bytes):
    pa_ti_size_t = matrix_bytes[:4]
    matrix_bytes = matrix_bytes[4:]

    sizeof_size_t = pa_ti_size_t[0]
    size_t_endianness = 'big' if pa_ti_size_t[1] else 'little'
    size_t_is_integral = not bool(pa_ti_size_t[2])
    size_t_is_signed = bool(pa_ti_size_t[3])

    rows = int.from_bytes(matrix_bytes[:sizeof_size_t], byteorder= size_t_endianness, signed=size_t_is_signed)
    matrix_bytes = matrix_bytes[sizeof_size_t:]

    cols = int.from_bytes(matrix_bytes[:sizeof_size_t], byteorder= size_t_endianness, signed=size_t_is_signed)
    matrix_bytes = matrix_bytes[sizeof_size_t:]

    pa_ti_T = matrix_bytes[:4]
    matrix_bytes = matrix_bytes[4:]

    sizeof_T = pa_ti_T[0]
    T_endianness = 'big' if pa_ti_T[1] else 'little'
    T_is_float = bool(pa_ti_T[2])
    
    assert T_is_float
    res = np.frombuffer(matrix_bytes, float_type(sizeof_T)).reshape(rows, cols)
    return res        
