from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum
from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Union
import math


class WeightInit(str, Enum):
    NORMAL = "NORMAL"
    KAIMING = "KAIMING"
    UNIFORM = "UNIFORM"
    

class ModelType(str, Enum):
    NANO_GPT = "NANO_GPT"
    TFL_GPT2 = "TRANSFORMERLENS_GPT2_SMALL"


class DatasetType(str, Enum):
    ABAB = 'abab'
    ADD = 'add'
    ALTERNATING = 'alternating'
    CYCLIC = 'cyclic'
    DIHEDRAL = 'dihedral'
    FLIPFLOP = 'flipflop'
    GRIDWORLD = 'gridworld'
    PARITY = 'parity'
    QUATERNION = 'quaternion'
    SYMMETRIC = 'symmetric'
    PERMUTATION_RESET = 'permutation_reset'


class OptimizerType(str, Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    

class DistributionType(str, Enum):
    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"


class ParameterisationType(str, Enum):
    """
    The parameterisation of the initialisation scales and learning rates.

    This only determines how the scales and learning rates are _scaled_ with the width of each part of the network!

    SP and PYTORCH do not apply any scaling to the learning rates, whereas MUP does.

    SP and PYTORCH differ in initialisation scaling of biases (see InitialisationConfig for more details).
    """
    MUP = "MUP"
    """mu-parameterisation"""
    SP = "SP"
    """Standard Parameterisation"""
    PYTORCH = "PYTORCH"
    """PyTorch default initialisation scaling (differs from SP in how bias initialisation is scaled)"""
    NONE = "NONE"
    """Keep default initialization and apply the init scale(s) in-place to each parameter"""
    
    
class InitialisationConfig(BaseModel):
    """
    Configuration for initialisation of the model parameters.

    To recover the default PyTorch initialisation, set:
     - default_init_scale = 1 / sqrt(3) ≈ 0.577
     - init_distribution = DistributionType.UNIFORM
     - parameterisation: ParameterisationType.PYTORCH

    (Note that the biases in the PYTORCH init. are scaled with 1 / sqrt(layer_fan_in), where the
    layer_fan_in above refers to the number of inputs to the layer after which biases are added.
    In Tensor Programs V, and in this repo, the fan_in of a bias is in contrast taken to be 1)
    (the default PyTorch initialisation initialises everything as Uniform(-1/sqrt(fan_in)), 1/sqrt(fan_in), where
    fan_in is the layer fan_in - number of inputs to the layer)

    Scale throughout refers to the "base" standard deviation of the distribution (and not e.g. bounds of the uniform).
    So, for example, when using Standard Parameterisation (SP) the standard deviation of the distribution to sample from
    for a weight matrix will be `scale / sqrt(fan_in)`
    """
    default_init_scale: float = Field(default=1.0, description="Default init scale if a param specific init_scale is not specified.")
    
    global_init_scale: float = Field(default=1.0, description="Multiplier to apply tor all init scales (including default)")
    
    init_scales_per_param: dict[str, float] = Field(default_factory=dict, description="If specified, overrides the default init_scale with a per-parameter init_scale")
    
    init_distribution: DistributionType = Field(default=DistributionType.NORMAL, description="The initialisation distribution")


class NanoGPTConfig(BaseModel):
    block_size: int = Field(default=1024, description="Max. sequence length")
    vocab_size: int = Field(default=50304, description="GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency")
    output_vocab_size: int = Field(default=None, description="Used for non-autoregressive case.")
    n_layers: int = 12
    n_heads: int = 12
    embed_dim: int = 768
    dropout: float = 0.0
    is_causal: bool = False
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # Init. settings (scales):
    # TODO
        
    
class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType = Field(default=OptimizerType.SGD)
    default_lr: float = Field(default=1e-3)
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    weight_decay: float = Field(default=0.0)
    clip_grad: float = Field(default=float("inf"))
    cosine_lr_schedule: bool = Field(default=False)

    @model_validator('default_lr')
    def check_lr(cls, v):
        if v <= 0:
            raise ValueError('Learning rate must be positive')
        return v


class DataLoaderConfig(BaseModel):
    """
    train_bs: Batch size for training.
    test_bs: Batch size for testing.
    num_workers: Number of workers to use.
    train_fraction: Fraction of dataset to be set aside for training.
    shuffle_train: Whether to shuffle the training data.
    seed: Random seed for reproducibility, ensuring function returns the same split with the same args.
    """
    train_bs: int = Field(default=64)
    test_bs: int = Field(default=32)
    num_workers: int = Field(default=1)
    train_fraction: float = Field(default=0.95)
    shuffle_train: bool = Field(default=True)
    

class DatasetConfig(BaseModel):
    """All automaton dataset classes below inherit from this."""
    dataset_filename: str = Field(default="filename.csv")
    data_folder: str = Field(default="data")
    dataset_type: str = Field(default=DatasetType)
    size: int = Field(1000)
    seed: int = Field(default=42)
    
    length: int = Field(default=20)
    random_length: bool = Field(default=False, description="Whether to use random length sequences, in which case take length attribute as max.")
            
    @validator('dataset_type')
    def check_dataset_type(cls, v):
        if not cls._validate_dataset_type(v):
            raise ValueError(f"Invalid dataset_type for {cls.__name__}")
        return v
    
    @classmethod
    def _validate_dataset_type(cls, dataset_type: str) -> bool:
        mapped_class = config_class_map.get(dataset_type)
        return mapped_class is cls
    
    @property
    @abstractmethod
    def output_vocab_size(self):
        """Abstract method to determine output vocabulary size of transformer."""
        pass
    
    
class BinaryInputAutomatonConfig(DatasetConfig):
    """Parent class for Parity, GridWorld, ABAB."""
    prob1: Optional[float] = Field(default=0.5, description="(float in [0,1]): probability of token 1")
    vocab_size: Optional[int] = Field(default=2, description="Vocab size of dataset (e.g. 2 for binary input)")


class ParityAutomatonConfig(BinaryInputAutomatonConfig): 
    @property
    def output_vocab_size(self):
        return 2
    
class GridworldAutomatonConfig(BinaryInputAutomatonConfig):
    class Label(str, Enum):
        STATE = "state"
        PARITY = "parity"
        BOUNDARY = "boundary"
    
    n: Optional[int] = Field(default=9, description="Number of states")
    label_type: Optional[Label] = Field(default=Label.STATE, description="-'state' (default): the state id, i.e. 0 to n-1.\n" \
            + "    - 'parity': the state id mod 2.\n" \
            + "    - 'boundary': whether the current state is in {0, n-1} or not.")
    
    @property
    def output_vocab_size(self):
        match self.label_type:
            case GridworldAutomatonConfig.Label.STATE: return self.n
            case GridworldAutomatonConfig.Label.PARITY | GridworldAutomatonConfig.Label.BOUNDARY: return 2


class ABABAutomatonConfig(BinaryInputAutomatonConfig):
    class Label(str, Enum):
        STATE = "state"
        BOUNDARY = "boundary"
    
    prob_abab_pos_sample: Optional[float] = Field(default=0.25, description="(float in [0,1]): probability of having a 'positive' sequence, i.e. 01010101010...")
    label_type: Optional[Label] = Field(default=Label.STATE, description="- 'state' (default): the state id.\n" \
            + "    - 'boundary': whether the state is in state 3 (the states are 0,1,2,3).")

    @property
    def output_vocab_size(self):
        match self.label_type:
            case ABABAutomatonConfig.Label.STATE: return 4 # Predict 0, 1, 2, 3 (see init of ABABAutomaton)
            case ABABAutomatonConfig.Label.BOUNDARY: return 2


class AdderAutomatonConfig(BinaryInputAutomatonConfig):
    class Label(str, Enum):
        STATE = "state"
        DIGIT = "digit"
        POSITION = "position"
    
    n_addends: Optional[int] = Field(default=2, description="Number of binary numbers to be added; default as 2.")
    label_type: Optional[Label] = Field(default=Label.STATE, description="choosing from the following options: \n" \
            +f"    - 'state': the state id, i.e. the int for the base-{n_addends} int corresponding to the number (carry, digit). \n" \
            +f"    - 'digit': the current output base-{n_addends} digit, without the carry. \n" \
            + "    - 'position': the current carry bit.")
    
    @property
    def output_vocab_size(self):
        match self.label_type:
            # Int for the base-{self.n_addends} int corresponding to the number (carry, digit).
            # Adding n_addends binary numbers so max sum at any position is 2**n - 1 (input all 1s) and carry can be at most 1. So total states is 2 * (2**self.n_addends - 1).
            case AdderAutomatonConfig.Label.STATE: return 2 * (2**self.n_addends -1)
            case AdderAutomatonConfig.Label.DIGIT: return self.n_addends # Current output base-{self.n_addends} digit, without the carry.
            case AdderAutomatonConfig.Label.POSITION: return 2 # Current carry bit.


class FlipFlopAutomatonConfig(DatasetConfig):
    n: Optional[int] = Field(default=2, description="Number of states")
    vocab_size: Optional[int] = Field(default=2, description="Vocab size of dataset (e.g. 2 for binary input)")
    
    @property
    def output_vocab_size(self):
        """For flip flop automaton the only output possibility is the state."""
        return self.n
    

class PermutationAutomatonConfig(DatasetConfig):
    """Parent class for Symmetric, Alternating (which directly takes this class config)."""
    class Label(str, Enum):
        """Types of labels possible for this class."""
        STATE = "state"
        FIRST_CHAIR = "first_chair"
    
    n: Optional[int] = Field(default=5, description="Symmetry group number.")
    label_type: Optional[Label] = Field(default=Label.STATE, description="- 'state' (default): the state id.\n" \
            + "    - 'first_chair': the element in the first position of the permutation.\n" \
            + "          e.g. if the current permutation is [2,1,4,3], then 'first_chair' is 2.")

    @property
    def output_vocab_size(self):
        match self.label_type:
            case PermutationAutomatonConfig.Label.STATE: return math.factorial(self.n) # Number of states for symmetry group size n
            case PermutationAutomatonConfig.Label.FIRST_CHAIR: return self.n # Number of unique labels for symmetry group
            

class SymmetricAutomatonConfig(PermutationAutomatonConfig):
    """Inherits from PermutationAutomaton class, including attributes:
    - n (int): number of objects, i.e. there are n! states.
    """
    n_actions: Optional[int] = Field(default=3, description="Number of permutations to include in the generator set, with 3 default actions: id, shift-by-1, swap-first-two)")

class AlternatingAutomatonConfig(PermutationAutomatonConfig):
    """Dummy class to show inheritance structure."""
    pass


class CyclicAutomatonConfig(DatasetConfig):
    n: Optional[int] = Field(default=5, description="Number of states")
    n_actions: Optional[int] = Field(default=2, description="Number of actions/generators, which shift by i positions, for i = 0 to n_actions-1.")
    
    @property
    def output_vocab_size(self):
        return self.n

## TODO: classes for Dihedral, Quaternion, PermutationReset

class RLCTSamplerType(str, Enum):
    SGLD = "SGLD"
    SGNHT = "SGNHT"


class SGNHT_Kwargs(BaseModel):
    lr: float
    diffusion_factor: float
    bounding_box_size: float
    num_samples: int


class SGLD_Kwargs(BaseModel):
    lr: float
    noise_level: float
    weight_decay: float
    elasticity: float
    temperature: str
    num_samples: int


class RLCTConfig(BaseModel):
    sampling_method: RLCTSamplerType
    sigma: float
    num_chains: int
    num_draws: int
    num_burnin_steps: int
    num_steps_bw_draws: int
    batch_size: int
    cores: int
    use_distill_loss: bool
    save_results: bool
    seed: Optional[Union[int, List[int]]] = Field(default=None)
    pbar: Optional[bool] = Field(default=True)
    verbose: Optional[bool] = Field(default=True)
    return_weights: Optional[bool] = Field(default=True)
    sgld_kwargs: Optional[SGLD_Kwargs] = Field(default=None)
    sgnht_kwargs: Optional[SGNHT_Kwargs] = Field(default=None)
    

class WandBConfig(BaseModel):
    log_to_wandb: bool = Field(default=True)
    save_model_as_artifact: bool = Field(default=True)
    model_save_path: str = Field(default="trained_models")
    wandb_project_name: str = Field(default="devinterp")
    sweep: bool = Field(default=False)
    
    def model_factory(self):
        return WandBConfig(
            log_to_wandb=True,
            save_model_as_artifact=True,
            wandb_project_name="devinterp-automata",
            sweep=False
        )


class MainConfig(BaseModel):
    ## Critical
    dataset_type: DatasetType
    model_type: ModelType
    
    ## Data
    dataset_config: DatasetConfig = Field(default_factory=DatasetConfig)
    adder_config: Optional[AdderAutomatonConfig] = Field(default_factory=AdderAutomatonConfig)
    abab_config: Optional[ABABAutomatonConfig] = Field(default_factory=ABABAutomatonConfig)
    alternating_config: Optional[AlternatingAutomatonConfig] = Field(default_factory=AlternatingAutomatonConfig)
    cyclic_config: Optional[CyclicAutomatonConfig] = Field(default_factory=CyclicAutomatonConfig)
    flipflop_config: Optional[FlipFlopAutomatonConfig] = Field(default_factory=FlipFlopAutomatonConfig)
    gridworld_config: Optional[GridworldAutomatonConfig] = Field(default_factory=GridworldAutomatonConfig)
    parity_config: Optional[ParityAutomatonConfig] = Field(default_factory=ParityAutomatonConfig)
    symmetric_config: Optional[SymmetricAutomatonConfig] = Field(default_factory=SymmetricAutomatonConfig)
    # TODO: permutation reset, dihedral, quaternion
    
    dataloader_config: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    
    ## Models
    nano_gpt_config: Optional[NanoGPTConfig] = Field(default=None)
    
    rlct_config: Optional[RLCTConfig] = None
    
    wandb_config: WandBConfig = Field(default_factory=WandBConfig.model_factory())
    
    ## Training bits and bobs
    num_training_iter: int = Field(default=10000)
    eval_frequency: Optional[int] = Field(default=None)
    num_eval_batches: Optional[int] = Field(default=20)
    loss_threshold: float = Field(default=1e-5)
        
    @property
    def num_epochs(self):
        """
        Getter for number of epochs.
        
        Epochs defined a bit differently: an epoch is the period of training in-between evaluations. This is to make it compatible with infinite trainloaders. If eval_frequency is None, then the two coincide.
        """
        if not self.eval_frequency: raise ValueError("eval_frequency must be set to get num_epochs")
        return math.ceil(self.num_training_iter / self.eval_frequency)
    
    @property
    def eval_frequency(self, train_loader: DataLoader):
        """Note evaluations occur during training."""
        return self.eval_frequency if self.eval_frequency is not None else len(train_loader)
    
    @property
    def run_name(self):
        pass # TODO: implement
    
    # TODO: check if you want logger to be in separate config class (start off with no for now)
    @property
    def is_wandb_enabled(self):
        """Whether wandb is enabled."""
        return self.wandb_config and self.wandb_config.log_to_wandb
    
    # Custom validator to adjust NanoGPTConfig based on DatasetConfig
    @validator('nano_gpt_config', pre=True, always=True)
    def adjust_nano_gpt_config(cls, nano_gpt_config, values):
        assert dataset_config.output_vocab_size is not None, "Output vocab size must be specified."
        dataset_config, dataset_type = values.get('dataset_config'), values.get('dataset_type')
        specific_dataset_config = values.get(f"{dataset_type}_config")
        if nano_gpt_config and dataset_config:
            nano_gpt_config.block_size = dataset_config.length
            nano_gpt_config.output_vocab_size = specific_dataset_config.output_vocab_size
        return nano_gpt_config
    

config_class_map = {
    "abab": ABABAutomatonConfig,
    "add": CyclicAutomatonConfig,
    "cyclic": CyclicAutomatonConfig,
    # TODO: dihedral
    "flipflop": FlipFlopAutomatonConfig,
    "gridworld": GridworldAutomatonConfig,
    "parity": ParityAutomatonConfig,
    # TODO "quaternion": QuaternionAutomatonConfig,
    # TODO: "permutation_reset": PermutationResetAutomatonConfig,
    "symmetric": SymmetricAutomatonConfig,
}
