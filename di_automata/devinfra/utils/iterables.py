import hashlib
import json
import warnings
from typing import (Any, Callable, Generator, Iterable, List, Tuple, TypeVar,
                    Union)

import numpy as np

T = TypeVar("T")


def to_tuple(x: Union[Tuple[int, ...], int]) -> Tuple[int, ...]:
    """
    Converts an integer or a tuple of integers to a tuple of integers.
    """
    if isinstance(x, int):
        return (x,)
    else:
        return x


def int_logspace(start, stop, num, return_type="list"):
    """
    Returns a set of integers evenly spaced on a log scale.
    """
    result = set(int(i) for i in np.logspace(np.log10(start), np.log10(stop), num))

    if len(result) != num:
        warnings.warn(
            f"Number of steps in int_logspace is not {num}, got {len(result)}."
        )

    if return_type == "set":
        return set(result)

    result = sorted(list(result))

    if return_type == "list":
        return result
    elif return_type == "np":
        return np.array(result)
    else:
        raise ValueError(
            f"return_type must be either 'list' or 'set', got {return_type}"
        )


def int_linspace(start, stop, num, return_type="list"):
    """
    Returns a set of integers evenly spaced on a linear scale.
    """
    result = set(int(i) for i in np.linspace(start, stop, num))

    if len(result) != num:
        warnings.warn(
            f"Number of steps in int_linspace is not {num}, got {len(result)}."
        )

    if return_type == "set":
        return result

    result = sorted(list(result))

    if return_type == "list":
        return list(result)
    elif return_type == "np":
        return np.array(result)
    else:
        raise ValueError(
            f"return_type must be either 'list' or 'set', got {return_type}"
        )


def map_nested(f, x):
    """Recursively applies a function to a nested dictionary or list."""
    if isinstance(x, dict):
        return {k: map_nested(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [map_nested(f, v) for v in x]
    else:
        return f(x)



def nested_update(d1: dict, d2: dict):
    """
    Updates the values in d1 with the values in d2, recursively.
    """
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = nested_update(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1


def flatten_dict(dict_, prefix="", delimiter="/", flatten_lists=False):
    """
    Recursively flattens a nested dictionary of metrics into a single-level dictionary.

    Parameters:
        dict_ (dict): The dictionary to flatten. It can contain nested dictionaries and lists.
        prefix (str, optional): A string prefix to prepend to the keys in the flattened dictionary.
                                 This is used internally for the recursion and should not typically
                                 be set by the caller.
        delimiter (str, optional): The delimiter to use between keys in the flattened dictionary.
        flatten_lists (bool, optional): Whether to flatten lists in the dictionary. If True, list
                                        elements are treated as separate metrics.

    Returns:
        dict: A flattened dictionary where the keys are constructed by concatenating the keys from
              the original dictionary, separated by the specified delimiter.

    Example:
        Input:
            {
                "Train": {"Loss": "train_loss", "Accuracy": "train_accuracy"},
                "Test": {"Loss": "test_loss", "Details": {"Test/Accuracy": "test_accuracy"}},
                "List": [1, 2, [3, 4]]
            }

        Output (with flatten_lists=True):
            {
                'Train/Loss': 'train_loss',
                'Train/Accuracy': 'train_accuracy',
                'Test/Loss': 'test_loss',
                'Test/Details/Test/Accuracy': 'test_accuracy',
                'List/0': 1,
                'List/1': 2,
                'List/2/0': 3,
                'List/2/1': 4
            }
    """
    flattened = {}
    for key, value in dict_.items():
        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value,
                    prefix=f"{prefix}{key}{delimiter}",
                    delimiter=delimiter,
                    flatten_lists=flatten_lists,
                )
            )
        elif isinstance(value, list) and flatten_lists:
            for i, v in enumerate(value):
                if isinstance(v, (dict, list)):
                    flattened.update(
                        flatten_dict(
                            {str(i): v},
                            prefix=f"{prefix}{key}{delimiter}",
                            delimiter=delimiter,
                            flatten_lists=flatten_lists,
                        )
                    )
                else:
                    flattened[f"{prefix}{key}{delimiter}{i}"] = v
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened


def unflatten_dict(d, delimiter="."):
    """Unflatten a dictionary where nested keys are separated by sep"""
    out_dict = {}
    for key, value in d.items():
        keys = key.split(delimiter)
        temp = out_dict
        for k in keys[:-1]:
            temp = temp.setdefault(k, {})
        temp[keys[-1]] = value
    return out_dict


def rm_none_vals(obj: T) -> T:
    """
    Recursively remove None values from a dictionary or list.
    """
    if isinstance(obj, dict):
        return {k: rm_none_vals(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [rm_none_vals(v) for v in obj if v is not None]

    if obj is None:
        raise ValueError("None value found in non-iterable object.")

    return obj


def filter_objs(objs: Iterable[T], **filters: Any) -> Generator[T, None, None]:
    """
    Filter a list of objects or dictionaries based on nested filters.

    Parameters:
        objs (List[T]): List of dictionaries or objects.
        filters (Any): Nested key-value pairs to filter the list.
                       Looks in both attributes and dictionary values.

    Yields:
        T: Matched object or dictionary.

    Raises:
        ValueError: If no matches or multiple matches are found.
    """

    def match(config, filters):
        flat_filters = flatten_dict(filters, delimiter="/")

        for k, v in flat_filters.items():
            keys = k.split("/")
            temp = config
            for key in keys:
                if hasattr(temp, "__contains__") and key in temp:
                    temp = temp[key]
                elif hasattr(temp, key):
                    temp = getattr(temp, key)
                else:
                    return False
            if temp != v:
                return False
        return True

    for obj in objs:
        if match(obj, filters):
            yield obj


def find_obj(objs: List[T], **filters: Any) -> T:
    """
    Find and return a single object based on filters.

    Parameters:
        objs (List[T]): List of dictionaries or objects.
        filters (Any): Nested key-value pairs to filter the list.

    Returns:
        T: Matched object or dictionary, or None if not found.
    """
    return next(filter_objs(objs, **filters))


def find_unique_obj(objs: List[T], **filters: Any) -> T:
    """
    Find and return a single object based on filters. Requires that only one object matches.

    Parameters:
        objs (List[T]): List of dictionaries or objects.
        filters (Any): Nested key-value pairs to filter the list.

    Returns:
        T: Matched object or dictionary, or None if not found.
    """
    objs = list(filter_objs(objs, **filters))

    if len(objs) == 0:
        raise ValueError("No matches found")
    elif len(objs) > 1:
        raise ValueError("Multiple matches found")
    
    return objs[0]


def hash_dict(d: dict):
    sorted_dict_str = json.dumps(d, sort_keys=True)
    m = hashlib.sha256()
    m.update(sorted_dict_str.encode('utf-8'))
    return m.hexdigest()


def prepend_dict(d: dict, prefix: str, delimiter="."):
    return {f"{prefix}{delimiter}{k}": v for k, v in d.items()}


def get_nested_attr(obj, path):
    for p in path.split('.'):
        obj = getattr(obj, p)
    return obj


def dict_to_latex(d):
    def prettify(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int) and value > 1000:
            # Add commas
            return f"{value:,}"
        else:
            return str(value)

    return "$" + ", ".join([f"{k}={prettify(v)}" for k, v in d.items()]) + "$"


def dicts_to_latex(*dicts):
    return "\n".join([dict_to_latex(d) for d in dicts])


def dict_to_slug(d, delimiter="_", equal_sign=""):
    return delimiter.join([f"{k}{equal_sign}{v}" for k, v in d.items()])