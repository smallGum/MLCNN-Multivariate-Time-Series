"""
    Specify the arguments
"""

from utils._libs_ import prod

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Function to convert dictionary of lists to list of dictionaries of all combinations of listed variables. 
Example:
    list_of_param_dicts({'a': [1, 2], 'b': [3, 4]}) ---> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
"""
def list_of_param_dicts(param_dict):
    """
    Arguments:
        param_dict   -(dict) dictionary of parameters
    """
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]

# -------------------------------------------------------------------------------------------------------------------------------------------------
"""
Arguments class that fits models
"""
class Args():
    """
    Arguments:
        arg_dict   -(dict) dictionary of model parameters
    """
    def __init__(self, arg_dict):
        self.__dict__ = arg_dict