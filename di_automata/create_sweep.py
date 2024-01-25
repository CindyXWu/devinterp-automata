import yaml


def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def construct_sweep_config(main_config_name: str, sweep_config_name: str):
    main_config = load_config(f'configs/{main_config_name}.yaml')
    sweep_config = load_config(f'configs/{sweep_config_name}.yaml')
    default_config = {
        "method": "bayes",
        "metric": {"name": "train.loss", "goal": "minimize"},
        "name": f"lr_sweep_{main_config.get('dataset_type')}_{main_config.get('model_type')}",
        # "program": "run.py",
        # "command": ["${env}", "${interpreter}", "${program}", "--file_path", "${args_json_file}"],
        "parameters": {
            "default_lr": {
                "distribution": "uniform",
                "min": 1e-4,
                "max": 1e-1
            },
        },
        'early_terminate': {'type': 'hyperband', 'min_iter': 5}
    }
    
    # Merge sweep_config into default_config
    for key, value in sweep_config.items():
        if isinstance(value, dict) and isinstance(default_config.get(key), dict): # If the value for this key is a dictionary, merge it with the default
            default_config[key] = {**default_config[key], **value}
        else: # Otherwise, overwrite the default value with the value from sweep_config
            default_config[key] = value
            
    return default_config