from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from abc import abstractmethod
from typing import Any, Optional, List, Union, Tuple
import math

from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.optim.sgld_ma import SGLD_MA
    

class ModelType(str, Enum):
    NANO_GPT = "NANO_GPT"
    TF_LENS = "TF_LENS"


class DatasetType(str, Enum):
    ABAB = 'abab'
    ADDER = 'adder'
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
    ADAMW = "ADAMW"
    

class DistributionType(str, Enum):
    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"


class RLCTLossType(str, Enum):
    CE = "ce"
    DISTILL = "distill"
    
    
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
    
    init_scales_per_param: Optional[dict[str, float]] = Field(default_factory=dict, description="If specified, overrides the default init_scale with a per-parameter init_scale")
    
    init_distribution: DistributionType = Field(default=DistributionType.NORMAL, description="The initialisation distribution")


class NanoGPTConfig(BaseModel):
    block_size: int = Field(default=1024, description="Should be no less than maximum sequence length.")
    vocab_size: int = Field(default=50304, description="GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency.")
    output_vocab_size: int = Field(default=None, description="Used for non-autoregressive case.")
    n_layers: int = Field(default=12, description="This will vary through experiments.")
    n_heads: int = 8
    embed_dim: int = 512
    dropout: float = Field(default=0.1, description="Default value was found to stabilise training for certain datasets. See Appendix B of automata paper.")
    is_causal: bool = False
    bias: bool = Field(default=True, description="True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster.")
        

class HookedTransformerConfig(BaseModel):
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_ctx: int  = Field(default=514, description="Set by root validator based on task. For scratchpad training, if input context is N, this should be 2 * N + 2.")
    n_layers: int
    d_vocab: int  = Field(default=None, description="Set by root validator. For scratchpad training, if N is output vocab size, this should be 2 * N + 1.")
    d_vocab_out: int = Field(default=None, description="Output vocab size for algorithmic tasks. Set by root validator.")
    act_fn: str = Field(default="relu")
    
    
class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType = Field(default=OptimizerType.ADAM)
    default_lr: float = Field(default=1e-3)
    global_lr: float = Field(default=1.0, description=" Multiplier for all learning rates")
    final_lr: float = Field(default=1e-4, description="For custom LR scheduler, define final LR.")
    per_param_lr: Optional[dict[str, float]] = Field(default_factory=dict, description="If specified, overrides the default lr with a per-parameter lr")
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    weight_decay: float = Field(default=0.0)
    clip_grad: float = Field(default=float("inf"))
    cosine_lr_schedule: bool = Field(default=False)
    dropout: float = Field(default=0.0)
    
    
class DataLoaderConfig(BaseModel):
    """
    train_bs: Batch size for training.
    test_bs: Batch size for testing.
    num_workers: Number of workers to use.
    train_fraction: Fraction of dataset to be set aside for training.
    shuffle_train: Whether to shuffle the training data.
    """
    train_bs: int = Field(default=64)
    test_bs: int = Field(default=32)
    num_workers: int = Field(default=1)
    train_fraction: float = Field(default=0.95)
    shuffle_train: bool = Field(default=True)
    

class DatasetConfig(BaseModel):
    """All automaton dataset classes below inherit from this."""
    dataset_type: DatasetType = Field(default=DatasetType.PARITY)
    size: int = Field(600000)
    length: int = Field(default=100, description="Paper uses sequence length 100.") 
    random_length: Optional[bool] = Field(default=False, description="Whether to use random length sequences, in which case take length attribute as max.")
    seed: Optional[int] = Field(default=None)
    pad_token: Optional[int] = Field(default=1, description="If specified, pad sequences by this length.")
    # vocab_size: Optional[int] = Field(default=None, description="Set by root validator. Input vocab size of transformer.")
    # output_vocab_size: Optional[int] = Field(default=None, description="Set by root validator. Output vocab size of transformer.")
    
    @property
    def dataset_filename(self):
        return f"{self.dataset_type}_{self.size}_{self.length}_{self.random_length}"
    
    @property
    @abstractmethod
    def vocab_size(self):
        """Determine input vocab size of transformer."""
        pass
        
    @property
    @abstractmethod
    def output_vocab_size(self):
        """Abstract method to determine output vocabulary size of transformer."""
        pass
    
    
class BinaryInputAutomatonConfig(DatasetConfig):
    """Parent class for Parity, GridWorld, ABAB."""
    prob1: float = Field(default=0.5, description="(float in [0,1]): probability of token 1")


class ParityAutomatonConfig(BinaryInputAutomatonConfig):
    @property
    def vocab_size(self):
        return 2
     
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
    def vocab_size(self): # TODO: check
        return 2
    
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
    def vocab_size(self):
        return 2
    
    @property
    def output_vocab_size(self):
        match self.label_type:
            case ABABAutomatonConfig.Label.STATE: return 5 # Predict [0,...,4] (see ABABAutomaton)
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
    def vocab_size(self):
        """
        Can have more than 2 despite binary input due to:
        0: Represents no sum and no carry (all inputs are 0).
        1: Represents a sum of 1 with no carry (exactly one input is 1, and there is no carry from the previous position, or it's the result of a carry without any 1s in the current position).
        2: Represents a sum of 2 or a carry and one input being 1 (two inputs are 1, or one input and a carry from the previous position).
        3: Represents a sum of 3 (two inputs are 1, and there is also a carry from the previous position)
        """
        return 4
    
    @property
    def output_vocab_size(self):
        match self.label_type:
            # Int for the base-{self.n_addends} int corresponding to the number (carry, digit).
            # Adding n_addends binary numbers so max sum at any position is 2**n - 1 (input all 1s) and carry can be at most 1. So total states is 2 * (2**self.n_addends - 1).
            case AdderAutomatonConfig.Label.STATE: return 2 * (2**self.n_addends - 1)
            case AdderAutomatonConfig.Label.DIGIT: return self.n_addends # Current output base-{self.n_addends} digit, without the carry.
            case AdderAutomatonConfig.Label.POSITION: return 2 # Current carry bit.


class FlipFlopAutomatonConfig(DatasetConfig):
    n: Optional[int] = Field(default=2, description="Number of states")
    
    @property
    def vocab_size(self):
        return self.n + 1
    
    @property
    def output_vocab_size(self):
        """For flip flop automaton the only output possibility is the state."""
        return self.n + 1
    

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
    def vocab_size(self):
        return self.n

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
    """Dummy class to show inheritance structure from PermutationAutomatonConfig."""
    pass


class CyclicAutomatonConfig(DatasetConfig):
    n: Optional[int] = Field(default=5, description="Number of states")
    n_actions: Optional[int] = Field(default=2, description="Number of actions/generators, which shift by i positions, for i = 0 to n_actions-1.")
    
    @property
    def vocab_size(self):
        return self.n
    
    @property
    def output_vocab_size(self):
        return self.n


class DihedralAutomatonConfig(DatasetConfig):
    class Label(str, Enum):
        """Types of labels possible for this class."""
        STATE = "state"
        TOGGLE = "toggle"
        POSITION = "position"
        
    n: Optional[int] = Field(default=4, description="Size of the 'cycle'. There are 2n states considering also the toggle bit.")
    label_type: Optional[Label] = Field(default=Label.STATE, description="'state': the state id, i.e. considering both toggle and position. \n" \
            + "    - 'toggle': the toggle bit (in {0, 1}). \n" \
            + "    - 'position': the position on the n-cycle (in [n]).")

    @property
    def vocab_size(self):
        return 2
    
    @property
    def output_vocab_size(self):
        match self.label_type:
            case DihedralAutomatonConfig.Label.STATE: return self.n * 2 # Toggle 0,1 and state
            case DihedralAutomatonConfig.Label.TOGGLE: return 2 # Toggle bit in {0, 1}
            case DihedralAutomatonConfig.Label.POSITION: return self.n # Position on the n-cycle in [1,...n]
    

class QuaternionAutomatonConfig(DatasetConfig):
    """This class is a simple creature."""
    @property
    def vocab_size(self):
        return 4

    @property
    def output_vocab_size(self):
        return 8
    

class PermutationResetAutomatonConfig(DatasetConfig):
    """Input to automaton is an action, which can either be application of a generator or a reset to a particular state. 
    
    Generators modify the current state based on the permutation they represent.
    Reset action directly sets the current state to a specific permutation.
    """
    n: int = Field(default=4, description="Should take values 4 or 5.")
    generators: Any = Field(default=[[1,0,2,3,4], [4,0,1,2,3]], description="List of generators for permutation group.")
    perm_probs: Optional[list[float]] = Field(default=None, description="Probability of any of the generator lists from generators. If not specified, return uniform distribution via validator method.")
    
    @validator("generators", pre=True, always=True)
    def check_generators(cls, v, values):
        n = values.get("n")
        assert len(v[0]) == n, "Generators must be of length n."
        return v
        
    @validator("perm_probs", pre=True, always=True)
    def set_perm_probs(cls, v, values):
        """
        Args:
            v: perm_probs value.
            values: dictionary of class attribute values.
        """
        if v is None:
            generators = values.get("generators", [])
            n_generators = len(generators)
            uniform_prob = 1.0 / n_generators if n_generators > 0 else 0
            return [uniform_prob for _ in range(n_generators)]
        return v

    @property
    def vocab_size(self):
        """There is one reset action for each possible state. 
        The number of possible states is n! since there are n! possible permutations of n elements.
        
        Total input vocabulary size (number of unique actions the automaton can take) is sum of number of generators and number of reset actions.
        """
        return math.factorial(self.n) + len(self.generators)

    @property
    def output_vocab_size(self):
        """The output of the automaton is the state after an action is applied. 
        State is represented by a permutation of n elements.
        """
        return math.factorial(self.n)


class RLCTSamplerType(str, Enum):
    SGLD = "SGLD"
    SGLD_MA = "SGLD_MA"


class SGLD_Kwargs(BaseModel):
    lr: float
    noise_level: float = Field(default=1.0, description="Standard deviation for Gaussian noise added in SGLD. Value should be set to not dominate gradient norm.")
    weight_decay: float = Field(default=1e-6, description="Something like [1e-5, 1e-6, 1e-7].")
    elasticity: float = Field(default=1.0, description="Something like [1, 10, 100].")
    bounding_box_size: float = Field(default=None, description="If set, prevents LLC estimator chain from wandering too far.")
    temperature: str = Field(default="adaptive", description="If adaptive, calculate temperature using number of samples seen, given by num_samples.")
    num_samples: int
    mh_frequency: int = Field(default=20, description="How many steps to take between metropolis-hastings acceptance ratio calculations as a diagnostic for SGLD.")


class EssentialDynamicsConfig(BaseModel):
    batches_per_checkpoint: int = Field(default=500, description="Number of batches in fixed dataset subset to get essential dynamics logits from.")
    eval_frequency: int = Field(default=10, description="Essential dynamics evaluation occurs more often than other metric logging.")
    
    
class RLCTConfig(BaseModel):
    rlct_loss_type: RLCTLossType
    sampling_method: RLCTSamplerType = Field(default=None, description="Value in config only used if the sampler type is not specified when called locally in code. This allows multiple samplers to be used at once.")
    num_chains: int = Field(default=10)
    num_draws: int = Field(default=100)
    num_samples: int = Field(default=None, description="Total unique samples seen during RLCT sampling = min(unique datapoints in dataset as defined by task config, datapoints seen based on num_draws and num steps bw draws). Set by root validator.")
    num_burnin_steps: int = Field(default=0, description="NOT IMPLEMENTED YET.")
    num_steps_bw_draws: int = Field(default=1)
    
    # Meta
    cores: int = Field(default=1)
    seed: Optional[Union[int, List[int]]] = Field(default=[None], description="Can be int or list of ints. Example: 1234 or [1234, 5678]. Must be same length as number of chains.")
    
    # Flags
    verbose: Optional[bool] = Field(default=True, description="Set to False to disable tqdm in sampler.")
    online: bool = Field(default=False)
    use_distill_loss: bool
    use_diagnostics: bool = Field(default=True, description="Whether to include norm, WBIC, gradient and loss diagnostics for RLCT estimation.")
    
    # Other configs
    sgld_kwargs: Optional[SGLD_Kwargs]
    ed_config: Optional[EssentialDynamicsConfig]
    
    # Saving
    rlct_model_save_dir: Optional[str]
    rlct_data_dir: Optional[str]
    

class WandBConfig(BaseModel):
    log_to_wandb: bool = Field(default=True, description="Set to false if testing only.")
    wandb_project_name: str = Field(default="devinterp-automata")
    entity_name: str = Field(default="wu-cindyx", description="Either WandB username or name of team.")
    sweep: bool = Field(default=False, description="Whether to run a sweep.")
    sweep_num: Optional[int] = Field(default=None, description="Number of repeats per sweep.")
    wandb_run_name_suffix: Optional[str] = Field(default=None, description="Additional run name e.g. for temporary or test runs.")
                
    @validator('sweep_num', pre=True, always=True)
    def check_sweep_num(cls, value, values):
        if values.get('sweep') and value is None: raise ValueError("Must specify sweep_num if sweep is True.")
        return value


class EDPlotConfig(BaseModel):
    transitions: Optional[Tuple[int, int, int]] = Field(default=None, description="List of tuples (start, end, stage), representing the starting point, ending point, and a stage or condition of each unique phase identified from ED plot. These points should have units of native iteration as the step, not checkpoint index (to make it easier to compare with WandB graphs).")
    colors: Optional[List[Tuple[int]]] = Field(default=None, description="List of RGB tuples for each state.")
    num_pca_components: int = 3
    plot_caustic: bool = True
    figsize: Tuple[int, int] = (20, 6)
    marked_cusp_data: Optional[list[dict[str,int]]] = Field(default=None, description="Key: ['step','influence_start','influence_end'], values are ints defining these. Not important for plotting, but seems to mark cusps.")
    use_cache: bool = False
    num_sharp_points: int = 20
    num_vertices: int = 35
    osculate_start: int = Field(default=1, description="Number of checkpoints to skip before calculating first osculating circle.")
    osculate_end_offset: int = Field(default=0, description="Number of checkpoints between last osculating circle checkpoint and last checkpoint during training.")
    osculate_skip: int = Field(default=8, description="Number of checkpoints to skip in between calculating each osculating circle.")
    smoothing_sigma_early: int = 10
    smoothing_sigma_late: int = 60
    smoothing_late_boundary: int = 200
    show_vertex_influence: bool = False
    ed_folder: str = Field(default=None, description="Folder to save essential dynamics plots to.")
    

class MainConfig(BaseModel):
    model_type: ModelType
    
    ## Other config classes
    # Leave class type for task config open and instantiate properly in root validator
    task_config: dict
    rlct_config: Optional[RLCTConfig]
    wandb_config: WandBConfig = Field(default_factory=WandBConfig)
    dataloader_config: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    initialisation: InitialisationConfig = Field(default_factory=InitialisationConfig)
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    
    # Model/logit saving
    model_save_method: str = Field(default="aws", description="How to save model - [local, aws, wandb]")
    aws_bucket: str = Field(default="s3://automata-devinterp-01/my_checkpoints/", description="Bucket name for AWS S3.")
    
    # Models
    nano_gpt_config: Optional[NanoGPTConfig]
    tflens_config: Optional[HookedTransformerConfig]
    
    ## Training bits and bobs
    # Flags
    llc_train: bool = Field(default=True, description="Whether to calculate RLCT/local learning coefficient/lambda hat metric from SLT during training.")
    ed_train: bool = Field(default=True, description="Whether to calculate essential dynamics (logit PCA) metric from SLT.")
    use_ema: bool = Field(default=True, description="Whether to use exponential moving average of model parameters.")
    use_scratchpad: bool = Field(default=True, description="Whether to use scratchpad training with recency bias as in Liu et al 'Transformers Learn Shortcuts to Automata' https://arxiv.org/abs/2210.10749")
    
    # Other
    num_model_save_workers: Optional[int] = Field(default=1, description="Number of workers to use for saving model checkpoints.")
    ema_decay: float = Field(default=0.9, description="Decay factor for EMA.")
    parameterisation: ParameterisationType = Field(default=ParameterisationType.MUP)
    num_training_iter: int = Field(default=10000)
    num_eval_batches: Optional[int] = Field(default=20)
    early_stop_patience: Optional[int] = Field(default=10, description="Number of evaluation steps with no improvement in log loss before stopping.")
    early_stop_acc_threshold: Optional[float] = Field(default=99.9, description="Percent accuracy to immediately stop training at.")
    
    # Set by validator
    run_name: Optional[str]
    is_wandb_enabled: Optional[bool]
    num_epochs: Optional[int]
    eval_frequency: Optional[int] = Field(default=None, decription="Defines number of steps per epoch. Very important for essential dynamics that is on order of 5000x < training iterations for enough checkpoints.")
    
    @root_validator(pre=True)
    def _set_fields(cls, v: dict):
        """Note evaluations occur during training.
        Eval_frequency must be specified at run-time if using an iterable train_loader.
        Epochs defined a bit differently: an epoch is the period of training in-between evaluations. This is to make it compatible with infinite trainloaders. If eval_frequency is None, then the two coincide.
        
        Args:
            v (dict): Stores attributes of MainConfig object.
        """
        # Instantiate correct class for task config
        task_config = v.get("task_config")
        assert task_config is not None, "Task config not specified."
        
        config_class = config_class_map[task_config["dataset_type"]]
        task_config_instance = config_class(**task_config)
        # Add input and output vocab size to task_config!
        v["task_config"]["output_vocab_size"] = task_config_instance.output_vocab_size
        v["task_config"]["vocab_size"] = task_config_instance.vocab_size
        
        if not v["eval_frequency"]:
            v["eval_frequency"] = task_config["size"]
        # Note run dataset name matches that in the dataset specific config and loaded up in dataloder
        v["run_name"] = f"{v['task_config']['dataset_type']}_{v['model_type']}_LR{v['optimizer_config']['default_lr']}_its{v['num_training_iter']}_layers{v['nano_gpt_config']['n_layers']}_seqlen{task_config['length']}_nstates{task_config.get('n', None)}_prob1{task_config.get('prob1', None)}_nactions{task_config.get('n_actions', None)}"
        v["is_wandb_enabled"] = v["wandb_config"] and v["wandb_config"]["log_to_wandb"]
        v["num_epochs"] = math.ceil(v["num_training_iter"] / v["eval_frequency"])
        
        # Set cosine LR schedule final value
        v["optimizer_config"]["final_lr"] = float(v["optimizer_config"]["default_lr"]) / 10
        
        # Adjust NanoGPTConfig based on task_config
        if v.get("nano_gpt_config") and task_config:
            v["nano_gpt_config"]["block_size"] = task_config["length"] + 1 # Add one for adder class in case of carry
            # The next two are properties and instiated only when a Pydantic object is created - do not access through values directly
            # Necessary as we don't know which class task_config belongs to when instantiated, until we map via task_config_map to correct config, using dataset_type as key
            v["nano_gpt_config"]["vocab_size"] = task_config_instance.vocab_size if not v["use_scratchpad"] else 2 * task_config_instance.vocab_size + 2
            v["nano_gpt_config"]["output_vocab_size"] = task_config_instance.output_vocab_size if not v["use_scratchpad"] else task_config_instance.output_vocab_size * 2 + 1
        
        # Adjust HookedTransformer config based on task_config
        if v.get("tflens_config") and task_config:
            v["tflens_config"]["n_ctx"] = task_config_instance.length if not v["use_scratchpad"] else 2 * task_config_instance.length + 2
            v["tflens_config"]["d_vocab"] = task_config_instance.vocab_size if not v["use_scratchpad"] else task_config_instance.vocab_size * 2
            v["tflens_config"]["d_vocab_out"] = max(task_config_instance.output_vocab_size, task_config_instance.vocab_size) if not v["use_scratchpad"] else max(task_config_instance.output_vocab_size, task_config_instance.vocab_size) * 2 + 1
        
        # Adjust RLCT parameters
        rlct_config = v["rlct_config"]
        # Save folder name
        if not rlct_config["use_distill_loss"]:
            v["rlct_config"]["rlct_data_dir"] = "rlct_data"
        else:
            v["rlct_config"]["rlct_data_dir"] = "rlct_data_distill"
        # Set total number of unique samples seen (n): if this is not done it will break LLC estimator
        v["rlct_config"]["sgld_kwargs"]["num_samples"] = v["rlct_config"]["num_samples"] = min( (rlct_config["num_draws"] * rlct_config["num_steps_bw_draws"] + rlct_config["num_burnin_steps"]) * v["dataloader_config"]["train_bs"], task_config_instance.vocab_size**task_config_instance.length)
        # Burnin not implemented yet
        assert rlct_config["num_burnin_steps"] == 0, 'Burn-in is currently not implemented correctly, please set num_burnin_steps to 0.'
        
        return v


class PostRunSLTConfig(BaseModel):
    """Use for specifying which run to load from WandB when post-processing. All other config attributes will be loaded from the run itself."""
    ## Crucial
    model_type: ModelType
    dataset_type: DatasetType
    
    ## Other config classes
    task_config: dict
    ed_plot_config: EDPlotConfig = Field(default_factory=EDPlotConfig)
    rlct_config: Optional[RLCTConfig]
    
    ## Training bits and bobs
    lr: float
    num_training_iter: int
    n_layers: int
    n_heads: int
    seq_len: int
    truncate_its: Optional[int] = Field(default=None, description="Truncate training iterations to this number.")
    
    ## Early stopping/truncation/checkpointing
    early_stop_patience: Optional[int] = Field(default=5, description="Number of evaluation steps with no improvement in log loss before stopping.")
    early_stop_smoothing_window: Optional[int] = Field(default=5, description="Size of moving average window to smooth out loss.")
    early_stop_acc_threshold: Optional[float] = Field(default=99.9, description="Percent accuracy to immediately stop training at.")
    skip_cps: int = Field(default=1, description="How many logged checkpoints to skip between downloads for analysis.")
    
    ## Flags
    llc: bool = Field(default=False, description="Whether to calculate RLCT/local learning coefficient/lambda hat metric from SLT from checkpoints outside of training.")
    ed: bool = Field(default=True, description="Whether to calculate essential dynamics (logit PCA) metric from SLT from checkpoints outside of training.")
    osculating: bool = Field(default=True, description="Whether to make osculating circle plots.")
    form: bool = Field(default=False, description="Whether to plot form potential over time. Requires marked_cusp_data to be not None in EDPlotConfig.")
    use_logits: bool = Field(default=True, description="Whether to load logits directly for ED calculations.")
    save_logits_cp: bool = Field(default=True, description="Save logits from each checkpoint directly to WandB at ED frequency. Used because we save the model checkpoints less frequently (for now).")
    
    ## WandB
    wandb_project_name: str = Field(default="devinterp-automata")
    entity_name: str = Field(default="wu-cindyx", description="Either WandB username or name of team.")
    run_name: Optional[str] = Field(default=None, description="Set by validator.")
    run_idx: int = Field(default=0, description="Counting back from runs with the same name in time, which run do you want to select?")
    
    @root_validator(pre=True)
    def _set_fields(cls, v: dict):
        # Instantiate correct class for task config
        task_config = v["task_config"]
        config_class = config_class_map[task_config["dataset_type"]]
        
        ## For new run names
        # v["run_name"] = f"{v['task_config']['dataset_type']}_{v['model_type']}_LR{v['lr']}_its{v['num_training_iter']}_layers{v['n_layers']}_heads{v['n_heads']}seqlen{v['seq_len']}_nstates{task_config.get('n', None)}_prob1{task_config.get('prob1', None)}_sigmastart{v['ed_plot_config']['smoothing_sigma_early']}_sigmalate{v['ed_plot_config']['smoothing_sigma_late']}"
        v["run_name"] = f"{v['task_config']['dataset_type']}_{v['model_type']}_LR{v['lr']}_its{v['num_training_iter']}_layers{v['n_layers']}_seqlen{v['seq_len']}_nstates{task_config.get('n', None)}_prob1{task_config.get('prob1', None)}_nactions{task_config.get('n_actions', None)}"

        # Adjust RLCT parameters
        rlct_config = v["rlct_config"]
        # Set save folder name
        if not rlct_config["use_distill_loss"]:
            v["rlct_config"]["rlct_data_dir"] = "rlct_data"
        else:
            v["rlct_config"]["rlct_data_dir"] = "rlct_data_distill"
        # Burnin not implemented yet
        assert rlct_config["num_burnin_steps"] == 0, 'Burn-in is currently not implemented correctly, please set num_burnin_steps to 0.'
        
        return v
    
    
config_class_map = {
    "abab": ABABAutomatonConfig,
    "adder": AdderAutomatonConfig,
    "alternating": AlternatingAutomatonConfig,
    "cyclic": CyclicAutomatonConfig,
    "dihedral": DihedralAutomatonConfig,
    "flipflop": FlipFlopAutomatonConfig,
    "gridworld": GridworldAutomatonConfig,
    "parity": ParityAutomatonConfig,
    "quaternion": QuaternionAutomatonConfig,
    "permutation_reset": PermutationResetAutomatonConfig,
    "symmetric": SymmetricAutomatonConfig,
}


rlct_class_map = {
    "SGLD": SGLD,
    "SGLD_MA": SGLD_MA
}