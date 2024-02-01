import torch
from torch import Tensor
import torch.nn as nn


def predictive_kl_loss(x: Tensor, y: Tensor, teacher_model: nn.Module, student_model: nn.Module, temperature: float = 1., **kwargs) -> tuple[Tensor, Tensor]:
    """
    Standard knowledge distillation loss (equivalent to KL divergence between the tempered teacher and student softmax outputs).
    """
    with torch.no_grad():
        teacher_logits = teacher_model(x)
    student_logits = student_model(x)
    return (
        nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / temperature, dim=-1),
            nn.functional.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
            reduction="batchmean",
        ),
        student_logits,
    )