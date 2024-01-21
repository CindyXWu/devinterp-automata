import yaml
from dataclasses import dataclass, fields, Field, MISSING
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from config_setup import *

    
def dataclass_to_dict(dc: dataclass) -> Dict[str, Any]:
    """Called on by create_yaml_template."""
    def get_default_value(field: Field) -> Any:
        "Helper function: get value of a field in a dataclass, regardless of type."
        default_value = field.default
        if default_value == MISSING:
            return None
        if isinstance(default_value, Enum):
            return default_value.value
        if isinstance(default_value, dataclass):
            return dataclass_to_dict(default_value)
        return default_value
    return {f.name: get_default_value(f) for f in fields(dc)}


def create_yaml_template(config_class: dataclass, filename: str) -> None:
    """Turn dataclass into a YAML template."""
    filepath = (Path(__file__).resolve().parent / 'configs' / filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(dataclass_to_dict(config_class), f, default_flow_style=False)


if __name__ == "__main__":
    config_dict = create_yaml_template(MainConfig(), 'main_test')