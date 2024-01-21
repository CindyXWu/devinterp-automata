from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ComplexityMeasure(str, Enum):
    ENTTROPY = "ENTROPY"
    BOOLEAN = "BOOLEAN"
    LZ = "LEMPEL_ZIV"
    CSR = "CRITICAL_SAMPLE_RATIO"
    

class WeightInit(str, Enum):
    NORMAL = "NORMAL"
    KAIMING = "KAIMING"
    UNIFORM = "UNIFORM"
    

class ModelType(str, Enum):
    MLP = "MLP"
    RNN = "RNN"
    LSTM = "LSTM"
    CNN = "CNN"
    TR = "TRANSFORMER"


class DatasetType(str, Enum):
    PARITY = "ParityDataset"
    PALINDROME = "PalindromeDataset"
    PRIME = "PrimeDataset"
    

class OptimizerType(str, Enum):
    SGD = "SGD"
    ADAM = "ADAM"


@dataclass
class MLPArchitectureConfig:
    hidden_sizes: list[int]
    output_size: int
    weight_init: WeightInit = WeightInit.KAIMING
    add_bias: bool = True
    

@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.SGD
    default_lr = 1e-3
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    weight_decay: float = 0.0
    clip_grad: float = float("inf")
    cosine_lr_schedule: bool = False


@dataclass
class DataLoaderConfig:
    """
    train_fraction: Fraction of dataset to be set aside for training.
    batch_size: For both train and test.
    shuffle: Whether to shuffle the training data.
    seed: Random seed for reproducibility, ensuring fn returns same split with same args.
    """
    train_bs: int = 64
    test_bs: int = 32
    num_workers: int = 1
    train_fraction: float = 0.95
    shuffle_train: bool = True
    seed: int = 42
    

@dataclass
class DatasetConfig:
    dataset_filename = "filename.csv"
    data_folder: str = "data"
    input_length: int = 20
    dataset_size: int = 10000
    p_prob: Optional[float] = 0.5
    """p_prob used for palindrome dataset only."""
    input_col_name: str = "binary_string"
    label_col_name: str = "label"
    
    
@dataclass
class MainConfig:
    dataset_type: DatasetType
    model_type: ModelType
    num_training_iter: int = 100000
    """For the purposes of stopping training at a specified loss, this number serves to upper bound to prevent infinite training."""
    optimization: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    loss_threshold: float = 1e-5

    eval_frequency: Optional[int] = 20
    """How many iterations between evaluations. If None, assumed to be 1 epoch, if the dataset is not Iterable."""
    num_eval_batches: Optional[int] = 20
    """
    How many batches to evaluate on. If None, evaluate on the entire eval dataLoader.
    Note, this might result in infinite evaluation if the eval dataLoader is not finite.
    """
    
    mlp_config: Optional[MLPArchitectureConfig] = None
    
    log_to_wandb: bool = True
    save_model_as_artifact: bool = True
    wandb_project_name: str = "iib-fcnn" # Serves as base project name - model type and dataset also included
    model_save_path: str = "trained_models"
    sweep: bool = False
    
    def __post_init__(self):
        match self.dataset_type:
            case DatasetType.PARITY:
                self.dataset.label_col_name = "has_even_parity"
            case DatasetType.PALINDROME:
                self.dataset.label_col_name = "has_palindrome"
        self.dataset.dataset_filename = f"{self.dataset_type}_{self.dataset.input_length}_{self.dataset.dataset_size}"
        self.wandb_project_name = f"{self.wandb_project_name}_{self.dataset_type}_{self.model_type}"
            