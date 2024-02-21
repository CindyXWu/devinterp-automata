# Copyright 2020 The HuggingFace Datasets Authors.
# Copyright 2023 Bingbin Liu, Jordan Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import math
from sympy.combinatorics.permutations import Permutation
import datasets
import numpy as np
from copy import copy
from omegaconf import OmegaConf, ListConfig, DictConfig

# Check Python version
import sys
major, minor = sys.version_info[:2]
version = major + 0.1*minor
OLD_PY_VERSION = 1 if version < 3.8 else 0

_CITATION = """\
@article{liu2022transformers,
  title={Transformers learn shortcuts to automata},
  author={Liu, Bingbin and Ash, Jordan T and Goel, Surbhi and Krishnamurthy, Akshay and Zhang, Cyril},
  journal={arXiv preprint arXiv:2210.10749},
  year={2022}
}
"""

_DESCRIPTION = """\
Non-autoregressive automaton simulation datasets.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {}

    
class AutomatonDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = []
    
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(**kwargs)
        
        """
        Class instantion handled by this parent class (cursed).
        
        Params:
            config: task_config from main config.
        """
        self.dataset_map = {
            'abab': ABABAutomaton,
            'adder': AdderAutomaton,
            'alternating': AlternatingAutomaton,
            'cyclic': CyclicAutomaton,
            'dihedral': DihedralAutomaton,
            'flipflop': FlipFlopAutomaton,
            'gridworld': GridworldAutomaton,
            'parity': ParityAutomaton,
            'quaternion': QuaternionAutomaton,
            'symmetric': SymmetricAutomaton,
            'permutation_reset': PermutationResetAutomaton
            # TODO: add Dyck
        }
        self.data_config = config
        # Instantiated dataset class object
        self.automaton = self.dataset_map[self.data_config.dataset_type](self.data_config)

    def _info(self):
        features = datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value("int32"), length=-1),
                "label_ids": datasets.Sequence(datasets.Value("int32"), length=-1)
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            )
        ]

    # def _generate_examples(self):
    #     """Cindy note: I don't use this function in my code, but directly go for sample() in child class."""
    #     for i in itertools.count(start=0):
    #         if i == self.data_config.size:
    #             break
    #         x, y = self.automaton.sample()
    #         yield i, {
    #             "input_ids": x,
    #             "label_ids": y
    #         }


class Automaton:
    """
    This is a parent class that must be inherited.
    """
    def __init__(self, data_config: DictConfig):
        self.data_config = data_config

        if hasattr(self.data_config, 'seed') and data_config["seed"] is not None:
            self.np_rng = np.random.default_rng(self.data_config['seed'])
        else:
            self.np_rng = np.random.default_rng()
        self.T = self.data_config.length
        self.random_length = data_config.random_length
        self.__info__ = \
                "  - T (int): sequence length.\n" \
            + "  - random_length (int in {0, 1}): whether to randomly sample a length per sample.\n"

    def f(self, x):
        """
        Get output sequence given an input seq
        """
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def sample_length(self):
        if self.random_length:
            return self.np_rng.choice(range(1, self.T+1))
        return self.T

    def help(self):
            print(self.__info__)


class BinaryInputAutomaton(Automaton):
    """
    This is a parent class that must be inherited.
    
    Subclasses: ParityAutomaton, GridworldAutomaton, ABABAutomaton
    TODO: sample sequences with a given number of 1s
    """
    def __init__(self, data_config):
        super().__init__(data_config)

        self.prob1 = data_config['prob1']
        self.__info__ = "  - prob1 (float in [0,1]): probability of token 1\n" + self.__info__
            

    def f(self, x):
        raise NotImplementedError()

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.binomial(1, self.prob1, size=T)
        return x, self.f(x)


class ParityAutomaton(BinaryInputAutomaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.name = 'parity'

        self.__info__ = "Parity machine with 2 states: \n" \
            + "- Inputs: binary strings\n" \
            + "- Labels: binary strings of the partial parity\n" \
            + "- Config: \n" \
            + self.__info__

    def f(self, x):
        return np.cumsum(x) % 2


class GridworldAutomaton(BinaryInputAutomaton):
    """
    Note: gridworld currently doesn't include a no-op.
    """
    def __init__(self, data_config):
        super().__init__(data_config)
        """
        NOTE: n is the number of states, and S is the id (0-indexing) of the rightmost state.
            i.e. the states are 0,1,2,...,S, where S=n-1.
        """
        
        self.n = data_config['n'] 
        self.S = self.n - 1
        self.label_type = data_config['label_type']
        self.name = f'Grid{self.n}'

        self.__info__ = f"1d Gridworld of n={self.n} states:\n" \
            + "- Inputs: binary strings, i.e. move left(0) or right(1)\n" \
            + "- Labels: depending on 'label_type'. \n" \
            + "- Config: \n" \
            + "  - n (int): number of states; i.e. the states are 0,1,2,...,n-1.\n" \
            + "  - label_type (str): choosing from the following options:\n" \
            + "    - 'state' (default): the state id, i.e. 0 to n-1.\n" \
            + "    - 'parity': the state id mod 2.\n" \
            + "    - 'boundary': whether the current state is in {0, n-1} or not.\n" \
            + self.__info__

    def f(self, x):
        x = copy(x)
        x[x == 0] = -1
        if OLD_PY_VERSION:
            # NOTE: for Python 3.7 or below, accumulate doesn't have the 'initial' argument.
            x = np.concatenate([np.array([0]), x]).astype(np.int64)
            states = list(itertools.accumulate(x, lambda a,b: max(min(a+b, self.S), 0)))
            states = states[1:]
        else:
            states = list(itertools.accumulate(x, lambda a,b: max(min(a+b, self.S), 0), initial=0))
            states = states[1:] # remove the 1st entry with is the (meaningless) initial value 0
        return np.array(states).astype(np.int64)


class ABABAutomaton(BinaryInputAutomaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.name = 'abab'
        self.prob_abab_pos_sample = data_config['prob_abab_pos_sample']
        self.label_type = data_config['label_type']
        self.transition = np.array([
            [4, 1], # state 0
            [2, 4], # state 1
            [4, 3], # state 2
            [0, 4], # state 3
            [4, 4], # state 4
        ])

        self.__info__ = "abab: an automaton with 4 states + 1 absorbing state:\n" \
            + "- Inputs: binary strings\n" \
            + "- Labels: depending on 'label_type'.\n" \
            + "- Config:\n" \
            + "  - prob_abab_pos_sample (float in [0,1]): probability of having a 'positive' sequence, i.e. 01010101010...\n" \
            + "  - label_type (str): choosing from the following options:\n" \
            + "    - 'state' (default): the state id.\n" \
            + "    - 'boundary': whether the state is in state 3 (the states are 0,1,2,3).\n" \
            + self.__info__ 

    def f(self, x):
        labels = []
        curr_state = 3
        for each in x:
            curr_state = self.transition[curr_state, each]
            labels += curr_state,
        labels = np.array(labels).astype(np.int64)
        if self.label_type == 'boundary':
            labels = (labels == 3).astype(np.int64)
        return labels

    def sample(self):
        pos_sample = self.np_rng.random() < self.prob_abab_pos_sample
        if pos_sample:
            T = self.sample_length()
            x = [0,1,0,1] * (T//4)
            x += [0,1,0,1][:(T%4)]
            x = np.array(x)
            return x, self.f(x)
        else:
            return super().sample()


class AdderAutomaton(BinaryInputAutomaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.name = 'addition'
        self.n_addends = data_config["n_addends"]
        self.addend_scales = np.array([2**i for i in range(self.n_addends)]).reshape(-1, 1)
        self.label_type = data_config['label_type']

        self.__info__ = f'Adder of n={self.n_addends} binary numbers:\n' \
            +f"- Inputs: {self.n_addends} binary numbers, encoded as the int for the {self.n_addends}-bit binary number.\n" \
            + "- Labels: depending on the label_type.\n" \
            + "- Config:\n" \
            + "  - n_addends (int): number of binary numbers to be added; default as 2.\n" \
            + "  - label_type (str): choosing from the following options: \n" \
            +f"    - 'state': the state id, i.e. the int for the base-{self.n_addends} int corresponding to the number (carry, digit). \n" \
            +f"    - 'digit': the current output base-{self.n_addends} digit, without the carry. \n" \
            + "    - 'position': the current carry bit.\n" \
            + self.__info__

    def f(self, x):
        outputs, carries = [], []
        carry = 0
        T = x.shape[-1]
        for i in range(T):
            curr_sum = x[:, i].sum() + carry
            # NOTE: 'mod n_addends' makes sure the carry is binary
            output, carry = curr_sum % self.n_addends, curr_sum // self.n_addends
            outputs += output,
            carries += carry,
        outputs = np.array(outputs).astype(np.int64)
        carries = np.array(carries).astype(np.int64)

        if self.label_type == 'state':
            return outputs + self.n_addends*carries
        elif self.label_type ==  'digit':
            return outputs
        elif self.label_type == 'carry':
            return carries

    def sample_addend(self, T):
        a = self.np_rng.binomial(1, self.prob1, size=T)
        return a

    def sample(self):
        T = self.sample_length()
        x = np.stack([self.sample_addend(T) for _ in range(self.n_addends)])
        # Pad the most significant bit (rightmost position, i.e. we're reversing the number) with 0 to handle the potential carry
        pad = np.zeros((self.n_addends, 1))
        x = np.concatenate([x, pad], 1)

        x_encode = (self.addend_scales * x).sum(0)
        return x_encode, self.f(x)


class FlipFlopAutomaton(Automaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.name = 'flipflop'
        self.n_states = data_config['n'] 
        self.n_actions = self.n_states + 1
        self.transition = np.array([list(range(self.n_actions))] + [[i+1]*self.n_actions for i in range(self.n_states)]).T

        self.__info__ = f"Flipflop with n={self.n_states} states:\n" \
            +f"- Inputs: tokens are either 0 (read) or 1:{self.n_states} (write).\n" \
            + "- Labels: the state id.\n" \
            + "- Config:\n" \
            + "  - n (int): number of write states; i.e. the states are 1,2,...,n, plus a default start state 0.\n" \
            + self.__info__ 

    def f(self, x):
        state, states = 0, []
        for action_id in x:
            state = self.transition[state, action_id]
            states += state,
        return np.array(states).astype(np.float32)

    def sample(self):
        T = self.sample_length()
        rand = self.np_rng.uniform(size=T)
        nonzero_pos = (rand < 0.5).astype(np.int64)
        writes = self.np_rng.choice(range(1, self.n_states+1), size=T)
        x = writes * nonzero_pos
        return x, self.f(x)


class PermutationAutomaton(Automaton):
    """
    This is a parent class that must be inherited.
    Subclasses: SymmetricAutomaton, AlternatingAutomaton
    """
    def __init__(self, data_config):
        super().__init__(data_config) 
        
        self.n = data_config['n'] # the symmetric group Sn
        self.label_type = data_config['label_type']

        self.__info__ = \
            "  - label_type (str): choosing from the following options:\n" \
            + "    - 'state' (default): the state id.\n" \
            + "    - 'first_chair': the element in the first position of the permutation.\n" \
            + "          e.g. if the current permutation is [2,1,4,3], then 'first_chair' is 2.\n" \
            + self.__info__

    def get_state_label(self, state):
        enc = self.state_encode(state)
        return self.state_label_map[enc]

    def f(self, x):
        curr_state = np.arange(self.n)
        labels = []
        for action_id in x:
            curr_state = self.actions[action_id].dot(curr_state)

            if self.label_type == 'state':
                    labels += self.get_state_label(curr_state),
            elif self.label_type == 'first_chair':
                    labels += curr_state[0],
        return np.array(labels)

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), replace=True, size=T)

        return x, self.f(x)


class SymmetricAutomaton(PermutationAutomaton):
    """
    TODO: add options for labels as functions of states
    - parity (whether a state is even): this may need packages (e.g. Permutation from sympy)
    - position / toggle: for S3 ~ D6, we can add labels for substructures as in Dihedral groups.
    """
    def __init__(self, data_config):
        super().__init__(data_config)

        self.name = f'S{self.n}'

        """
        Get states
        """
        self.state_encode = lambda state: ''.join([str(int(each)) for each in state])
        self.state_label_map = {}
        for si, state in enumerate(itertools.permutations(range(self.n))):
            enc = self.state_encode(state)
            self.state_label_map[enc] = si

        """
        Get actions (3 defaults: id, shift-by-1, swap-first-two)
        """
        self.n_actions = data_config['n_actions']
        self.actions = {0: np.eye(self.n)}
        # shift all elements to the right by 1
        shift_idx = list(range(1, self.n)) + [0]
        self.actions[1] = np.eye(self.n)[shift_idx]
        # swap the first 2 elements
        shift_idx = [1, 0] + list(range(2, self.n))
        self.actions[2] = np.eye(self.n)[shift_idx]

        if self.n_actions > 3:
            # add permutations in the order given by itertools.permutations
            self.all_permutations = list(itertools.permutations(range(self.n)))[1:]
            cnt = 2
            for each in self.all_permutations:
                action = np.eye(self.n)[list(each)]
                if np.linalg.norm(action - self.actions[0]) == 0:
                    continue
                elif np.linalg.norm(action - self.actions[1]) == 0:
                    continue
                self.actions[cnt] = action
                cnt += 1
                if cnt == self.n_actions: break

        self.__info__ = f"Symmetric group on n={self.n} objects:\n" \
            +f"- Inputs: tokens are either 0 (no-op), or 1:{self.n_actions} (corresponding to {self.n_actions} permutations).\n" \
            + "- Labels: depending on 'label_type'.\n" \
            + "- Config:\n" \
            + "  - n (int): number of objects, i.e. there are n! states.\n" \
            + "  - n_actions (int): number of permutations to include in the generator set;\n" \
            + "        the ordering is given by itertools.permutations, and the first 'n_actions' permutations will be included.\n" \
            + self.__info__ 


class AlternatingAutomaton(PermutationAutomaton):
    """
    TODO: other choices of generators (currently using (12x))?
    
    Cindy note: this is the only dataset class which only inherits and doesn't include additional attributes in the config, so should not have a unique config class.
    """
    def __init__(self, data_config):
        super().__init__(data_config)

        self.name = f'A{self.n}'

        """
        Get states
        """
        self.state_label_map = {}
        self.state_encode = lambda state: ''.join([str(int(each)) for each in state])
        cnt = 0
        for si, state in enumerate(itertools.permutations(range(self.n))):
            if not Permutation(state).is_even:
                    continue
            enc = self.state_encode(state)
            self.state_label_map[enc] = cnt
            cnt += 1

        """
        Get actions: all 3 cycles of the form (12x)
        """
        self.actions = {0: np.eye(self.n)}
        for idx in range(2, self.n):
            # (1, 2, idx) 
            shift_idx = list(range(self.n))
            shift_idx[0],shift_idx[1], shift_idx[idx] = shift_idx[1], shift_idx[idx], shift_idx[0]
            self.actions[idx-1] = np.eye(self.n)[shift_idx]
        self.n_actions = len(self.actions)

        self.__info__ = f"Alternating group on n={self.n} objects:\n" \
            +f"- Inputs: tokens from 0 to n-3, corresponding to all 3-cycles of the form (12x).\n" \
            + "- Labels: depending on 'label_type'.\n" \
            + "- Config:\n" \
            + "  - n (int): number of objects, i.e. there are n!/2 states.\n" \
            + self.__info__ 


class CyclicAutomaton(Automaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.n = data_config['n']
        
        """
        Get actions: shift by i positions, for i = 0 to n_actions-1
        """
        self.n_actions = data_config['n_actions']
        shift_idx = list(range(1, self.n)) + [0]
        self.actions = {}
        for i in range(self.n_actions):
            shift_idx = list(range(i, self.n)) + list(range(0, i))
            self.actions[i] = np.eye(self.n)[shift_idx]

        self.__info__ = 'Cyclic group of n={self.n} states:\n' \
            +f"- Inputs: tokens from 0 to n_actions-1\n" \
            + "- Labels: the current state.\n" \
            + "- Config:\n" \
            + "  - n (int): number of states.\n" \
            + "  - n_actions (int): number of actions/generators, which are 0, 1, ..., n_actions-1.\n" \
            + self.__info__

    def f(self, x):
        return np.cumsum(x) % self.n

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), replace=True, size=T)

        return x, self.f(x)


class DihedralAutomaton(Automaton):
    def __init__(self, data_config):
        super().__init__(data_config)
        
        self.n = data_config['n']
        self.label_type = data_config['label_type']

        """
        2 actions: toggle, or shift by 1 position (direction determined by the toggle).
        """
        self.n_actions = 2
        self.actions = {}
        # shift all elements to the left (counter-clockwise) by 1
        shift_idx = list(range(1, self.n)) + [0]
        self.actions[0] = np.eye(self.n)[shift_idx]
        # shift all elements to the right (closewise) by 1
        shift_idx = [self.n-1] + list(range(self.n-1))
        self.actions[1] = np.eye(self.n)[shift_idx]

        self.__info__ = 'Dihedral group of order 2n, where n={self.n}:\n' \
            +f"- Inputs: binary tokens:\n" \
            + "      0 for toggle, i.e. change direction in the n-cycle;\n" \
            + "      1 for drive,  i.e. move forward 1 step on the n-cycle.\n" \
            + "- Labels: depending on the label_type.\n" \
            + "- Config:\n" \
            + "  - n (int): size of the 'cycle'; i.e. there are 2n states considering also the toggle bit.\n" \
            + "  - label_type (str): choosing from the following options: \n" \
            + "    - 'state': the state id, i.e. considering both toggle and position. \n" \
            + "    - 'toggle': the toggle bit (in {0, 1}). \n" \
            + "    - 'position': the position on the n-cycle (in [n]).\n" \
            + self.__info__

    def f_sequential(self, x):
        # sanity check: sequential solution
        position = np.arange(self.n)
        states = []
        toggle = 0 # i.e. parity
        for action in x:
            if action == 0:
                # toggle direction
                toggle = 1 - toggle
            else:
                # drive by 1
                position = self.actions[toggle].dot(position)
            states += (toggle, position[0]),
        return states

    def f(self, x):
        # Parallel solution

        # Get toggles: a parity task on the toggle bit
        toggles = (x == 0).astype(np.int64)
        toggle_status = np.cumsum(toggles) % 2

        # Get positions: a directed modular counter
        directions = (-1)**toggle_status
        directed_drives = (x != 0).astype(np.int64) * directions
        positions = np.cumsum(directed_drives) % self.n

        if self.label_type == 'state':
            labels = [self.get_state_label(each) for each in zip(toggle_status, positions)]
            return np.array(labels).astype(np.int64)
        elif self.label_type == 'toggle':
            return toggle_status
        elif self.label_type == 'positions':
            return positions

    def get_state_label(self, state):
        """
        toggle in {0,1}
        position in [k]
        """
        toggle, position = state
        label = self.n*toggle + position
        return label

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), replace=True, size=T)

        return x, self.f(x)


class QuaternionAutomaton(Automaton):
    def __init__(self, data_config):
        super().__init__(data_config)

        self.n_states = 8 # {-1, 1} x {1, i, j, k}
        self.n_actions = 4  # {1, i, j, k}
        self.transition_pos = [
            0, 1, 2, 3,
            1, 4, 3, 6,
            2, 7, 4, 1,
            3, 2, 5, 4,
        ]
        self.transition_neg = [(each+4)%8 for each in self.transition_pos]
        self.transition = np.array(self.transition_pos + self.transition_neg)
        self.transition = self.transition.reshape(-1, 4)

        self.__info__ = "Quaternion group:\n" \
            + "- Inputs: tokens in {0,1,2,3}, corresponding to 1,i,j,k.\n" \
            + "- Labels: the state id; 8 states in total: 2 signs ({-1,1}) x 4 values ({1,i,j,k}).\n" \
            + "- Config:\n" \
            + self.__info__

    def f(self, x):
        curr_state = 0
        states = []
        for action_id in x:
            curr_state = self.transition[curr_state, action_id]
            states += curr_state,
        return np.array(states).astype(np.int64)

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_actions), size=T)
        return x, self.f(x)


class PermutationResetAutomaton(Automaton):
    def __init__(self, data_config):
        super().__init__(data_config)

        self.n = data_config['n']
        self.generators = data_config['generators']
        self.perm_probs = data_config['perm_probs']
        
        if type(self.generators[0]) is str:
            self.generators = [ np.array(list(map(int, list(g)))) for g in self.generators ]
        # Cast generators from ListConfig object to numpy array for indexing (hack for using Hydra)
        if isinstance(self.generators, (ListConfig, DictConfig)):
            self.generators = np.array(OmegaConf.to_container(self.generators, structured_config_mode="DICT"))

        self.n_states = math.factorial(self.n) # States = permutations; maybe rename
        self.n_generators = len(self.generators) # Actions = generators
        self.n_actions = self.n_states + self.n_generators # 1 reset symbol per state, 1 apply symbol per generator

        self.init_state = np.arange(self.n) # Identity permutation
        
        # Lookup tables
        self.int2perm = list(map(np.array, itertools.permutations(range(self.n))))
        self.perm2int = {tuple(p):i for i,p in enumerate(self.int2perm)}
        
        # Interval lengths
        T = self.sample_length()
        self.lags = [1]
        while self.lags[-1]*2 < T:
            self.lags.append(self.lags[-1]*2)

    def f(self, x):
        curr_state = self.init_state
        states = []
        for action_id in x:
            if action_id >= self.n_states:
                curr_state = self.generators[action_id - self.n_states][curr_state]
            else:
                curr_state = self.int2perm[action_id]
            states.append(self.perm2int[tuple(curr_state)])
        return np.array(states, dtype=np.int64)

    def sample(self):
        T = self.sample_length()
        x = self.np_rng.choice(range(self.n_generators), p=self.perm_probs, size=T) + self.n_states
        
        i = 0
        while i < T:
            x[i] = self.np_rng.choice(range(self.n_states))
            i += self.np_rng.choice(self.lags)
        
        return x, self.f(x)