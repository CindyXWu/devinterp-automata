import os
from contextlib import contextmanager
from typing import Literal, Optional, Union

import torch

DeviceLiteral = Literal["cpu", "cuda", "xla", "mps"]
DeviceOrDeviceLiteral = Union[torch.device, DeviceLiteral]


def get_default_device():
    """
    Returns the default device for PyTorch.
    """
    device = os.environ.get("DEVICE", None)

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla
        return torch.device("xla")
    except ModuleNotFoundError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_of(obj) -> Optional[DeviceOrDeviceLiteral]:
    """
    Returns the device of the given object. Returns the first device found.
    """
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            device = get_device_of(item)

            if device is not None:
                return device
    elif isinstance(obj, dict):
       for value in obj.values():
            device = get_device_of(value)

            if device is not None:
                return device
    elif hasattr(obj, "device"):
        return obj.device

    return None


def move_to_(obj, device: DeviceOrDeviceLiteral = "cpu"):
    """
    Moves the given object to the given device.
    """
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            move_to_(item, device)
    elif isinstance(obj, dict):
       for value in obj.values():
           move_to_(value, device)
    elif hasattr(obj, "to"):
        obj.to(device)

@contextmanager
def temporarily_move_to(obj, device: DeviceOrDeviceLiteral = "cpu"):
    """
    Temporarily moves the given object to the given device.
    """
    original_device = get_device_of(obj)
    print(original_device)

    if original_device is None:
        raise ValueError(f"Could not find device of {obj}")

    move_to_(obj, device)

    yield

    move_to_(obj, original_device)
