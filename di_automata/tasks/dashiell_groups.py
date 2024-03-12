import torch
from copy import deepcopy
from functools import reduce
from itertools import product
import math
import numpy as np
from operator import mul
import torch
from itertools import accumulate


class DihedralElement:
    def __init__(self, rot: int, ref: int, n: int):
        """
        self.rot: number of times r applied, mod n.
        self.ref: number of times s applied, mod 2.
        """
        self.n = n
        self.rot = rot % n
        self.ref = ref % 2

    def __repr__(self):
        return str((self.rot, self.ref))

    def __hash__(self):
        return hash(str(self))

    @property
    def sigma(self):
        return self.rot, self.ref

    @classmethod
    def full_group(cls, n):
        return [
            DihedralElement(r, p, n) for r, p in product(range(n), [0, 1])
        ]

    @property
    def order(self):
        if self.ref:
            return 2
        elif self.rot == 0:
            return 0
        elif (self.n % self.rot) == 0:
            return self.n // self.rot
        else:
            return math.lcm(self.n, self.rot) // self.rot

    @property
    def inverse(self):
        if self.ref:
            return deepcopy(self)
        else:
            return DihedralElement(self.n - self.rot, self.ref, self.n)

    def __mul__(self, other):
        if (not isinstance(other, DihedralElement)) or (other.n != self.n):
            raise ValueError(
                'Can only multiply a dihedral rotation with another dihedral rotation'
            )
        if self.ref:
            rot = self.rot - other.rot
        else:
            rot = self.rot + other.rot
        return DihedralElement(rot, self.ref + other.ref, self.n)

    def __pow__(self, x: int):
        if x == -1:
            return self.inverse
        elif x == 0:
            return DihedralElement(0, 0, self.n)
        elif x == 1:
            return deepcopy(self)
        else:
            return reduce(mul, [deepcopy(self) for _ in range(x)])

    def index(self) -> int:
        return self.rot * self.ref + self.rot


def generate_subgroup(generators, n):
    group_size = 0
    all_elements = set(generators)
    while group_size < len(all_elements):
        group_size = len(all_elements)
        rotations = [DihedralElement(*p, n) for p in all_elements]
        for r1, r2 in product(rotations, repeat=2):
            r3 = r1 * r2
            all_elements.add(r3.sigma)
    return list(all_elements)


def dihedral_conjugacy_classes(n: int):
    """n: order of cyclic group in dihedral."""
    conj_classes = [(0, 0), (0, 1)]
    if n % 2 == 1:
        m = (n - 1) // 2
    else:
        m = (n - 2) // 2
        conj_classes += [((2, 0), (0, 1)), ((2, 1), (1, 1))]
    conj_classes += [(i, 0) for i in range(1, m+1)]
    return conj_classes


class DihedralIrrep:

    def __init__(self, n: int, conjugacy_class):
        self.n = n
        self.conjugacy_class = conjugacy_class
        self.group = DihedralElement.full_group(n)
        if isinstance(conjugacy_class[0], int) and conjugacy_class[1] == 0 and self.conjugacy_class[0] != 0:
            self.dim = 2
        else:
            self.dim = 1
        
    def _trivial_irrep(self):
        return {r.sigma: torch.ones((1,)) for r in self.group}
    
    def _reflection_irrep(self):
        return {r.sigma: ((-1)**r.ref) * torch.ones((1,)) for r in self.group}
    
    def _subgroup_irrep(self, sg):
        return {r.sigma: 1 if r.sigma in sg else -1 for r in self.group}
    
    def _matrix_irrep(self, k):
        mats = {}
        mats[(0, 0)] = torch.eye(2)
        ref = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
        mats[(0, 1)] = ref
        two_pi_n = 2 * torch.pi / self.n
        for idx in range(1, self.n):
            sin = np.sin(two_pi_n * idx * k)
            cos = np.cos(two_pi_n * idx * k)
            m = torch.tensor([[cos, -1.0 * sin], [sin, cos]], dtype=torch.float32)
            mats[(idx, 0)] = m
            mats[(idx, 1)] = ref @ m
        return mats
    
    def matrix_representations(self):
        if self.conjugacy_class == (0, 0):
            return self._trivial_irrep()
        elif self.conjugacy_class == (0, 1):
            return self._reflection_irrep()
        elif self.conjugacy_class[1] == 0:
            return self._matrix_irrep(self.conjugacy_class[0])
        elif (
            isinstance(self.conjugacy_class[0], tuple) and 
            isinstance(self.conjugacy_class[1], tuple)
        ):
            subgroup = generate_subgroup(self.conjugacy_class, self.n)
            return self._subgroup_irrep(subgroup)
        else:
            raise ValueError(
                f'Somehow {self.conjugacy_class} is not a proper conjugacy class....'
            )
    
    def tensors(self):
        reps = self.matrix_representations()
        matrices = [reps[g.sigma] for g in self.group]
        if self.dim == 1:
            #tensor = torch.ones((len(self.group), 1), dtype=torch.float32)
            return torch.asarray(matrices).unsqueeze(1)
        return torch.stack(
            [
                torch.flatten(mat) for mat in matrices
            ],
            dim=0
        )


class ProductDihedralIrrep:

    def __init__(self, n: int, m: int, conjugacy_classes):

        assert len(conjugacy_classes) == m

        self.m = m
        self.n = n
        self.conjugacy_class = conjugacy_classes
        self.group = [element for element in product(DihedralElement.full_group(n), repeat=m)]
        self.irreps = [
            DihedralIrrep(n, conj_class).matrix_representations() for conj_class in conjugacy_classes
        ]

    def matrix_representations(self):

        reps = {}

        for elements in self.group:
            key = tuple(el.sigma for el in elements)
            matrices = [self.irreps[i][el] for i, el in enumerate(key)]
            reps[key] = reduce(torch.kron, matrices)
        return reps


def actions_to_labels(tensor, translation: dict, dtype: str = "float32"):
    actions = {
        0: DihedralElement(0, 1, 5),
        1: DihedralElement(1, 0, 5)
    }
    action_list = []
    for i in tensor:
        if i ==  torch.tensor(0):
            action_list.append(actions[0])
        else:
            action_list.append(actions[1])
    states = accumulate(action_list, mul)
    dtype = torch.float32 if dtype == "float" else torch.int32
    return torch.tensor([translation[s.sigma] for s in states], dtype=dtype)


def get_all_bits(m):
    """Generate every possible m-length binary sequence."""
    d = np.arange(2 ** m)
    bits = (((d[:,None] & (1 << np.arange(m)))) > 0).astype(int)
    return torch.asarray(bits)


def dihedral_fourier(fn_values, n):
    irreps = {conj: DihedralIrrep(n, conj).tensors() for conj in dihedral_conjugacy_classes(n)}
    return {conj: (fn_values * irrep).sum(dim=0) for conj, irrep in irreps.items()}


def get_fourier_spectrum(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    form: bool = False
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Outer loop is for label values and inner loop stores list of predicted values."""
    transforms = []
    means = []
    for i in range(10):
        mask = labels == i
        # Caution: applying the mask across multiple dimensions flattens logits
        print(type(logits[mask]))
        mean = logits[mask].mean(dim=0)
        mean = mean[torch.tensor([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])]
        transform = dihedral_fourier(mean.to('cpu').unsqueeze(1), 5)
        print(f"Label at {i}\n{transform}")
        transforms.append(transform)
        means.append(mean)
    return transforms, means


def analyse_power(transforms: list[np.ndarray], means: list[torch.Tensor]) -> None:
    total_power = 0.

    for i in range(len(transforms)):
        print(f"Label at {i}")
        print(means[i].pow(2).mean())
        
        for k, v in transforms[i].items():
            if len(v) == 1:
                power = (v**2).item()
            else:
                mat = v.reshape(2, 2)
                power = 2 * (mat @ mat.T).trace().item() 
            total_power += power
            print(f'{k}: {power / 100}') # Divide by 100 as this is 10^2 