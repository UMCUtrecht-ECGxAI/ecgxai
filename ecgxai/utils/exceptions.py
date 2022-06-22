"""
Exceptions.

MB Vessies
"""

class ModelOutputShapeMismatchException(Exception):
    def __init__(self, shape : tuple = None, correct_shape : tuple = None, message : str = None):
        if not message:
            message = f'Model output shape {shape} does not match expected output shape {correct_shape}.'
        super().__init__(message)


class SumMeanReductionShapeMismatchException(Exception):
    def __init__(self, dim : int = None, message : str = None):
        if not message:
            message = f'Loss reduction "sum_mean" requires a 3 dimenstional output, instead an output of shape {dim} was given.'
        super().__init__(message)


class IncorrectNumberOfTMWLabelsException(Exception):
    def __init__(self, metric : object, expected : int, recieved : int, message : str = None):
        if not message:
            message = f'Wrapped metric of type {type(metric).__name__} has an output of size {expected} but {recieved} labels were given.'
        super().__init__(message)
