"""
A Transformer model implementation, stol..argh borrowed from karphathy's nanoGPT
References:
https://github.com/karpathy/nanoGPT/blob/master/model.py

I have in turn borrowed this from Bruno's implementation of a transformer-classifier from the mechanistic-distillation codebase.
"""

from collections import OrderedDict
from enum import Enum
import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from torch import LongTensor, Tensor
from torch.nn import functional as F
from architectures.common import Identity, Lambda, SplitAggregate, Residual
from di_automata.config_setup import NanoGPTConfig


@torch.jit.script  # good to enable when not using torch.compile, disable when using
def new_gelu(x: Tensor):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        dropout: float,
        bias: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.in_projection = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.out_projection = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.embd_dim = embed_dim
        self.dropout = dropout
        self.is_causal = is_causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embd_dim)
        assert C == self.embd_dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.in_projection(x).split(
            self.embd_dim, dim=-1
        )  # (B, T, n_embd) -> 3 * (B, T, embd_dim)
        
        # Get in form (B, num_heads, T, head_dim)
        k = rearrange(k, 'b t (nh hd) -> b nh t hd', nh=self.n_heads)
        q = rearrange(q, 'b t (nh hd) -> b nh t hd', nh=self.n_heads)
        v = rearrange(v, 'b t (nh hd) -> b nh t hd', nh=self.n_heads)

        # Scaling for the attention logits (from Tensor Programs V: https://arxiv.org/pdf/2203.03466.pdf)
        q = q / math.sqrt(float(self.embd_dim // self.n_heads))

        # Causal self-attention; self-attend: (B, num_heads, T, head_dim) x (B, num_heads, T, head_dim) -> (B, num_heads, T, T)
        # Efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal,
        )
        # y = (
        #     y.transpose(1, 2).contiguous().view(B, T, C)
        # )  # re-assemble all head outputs side by side
        y = rearrange(y, 'b h t c -> b t (h c)')

        # output projection
        y = self.resid_dropout(self.out_projection(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self, n_heads: int, embed_dim: int, dropout: float, bias: bool, is_causal: bool = False
    ):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(
            n_heads=n_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            bias=bias,
            is_causal=is_causal,
        )
        self.ln_2 = LayerNorm(embed_dim, bias=bias)
        self.mlp = nn.Sequential(
            OrderedDict(
                c_fc=nn.Linear(embed_dim, 4 * embed_dim, bias=bias),
                gelu=Lambda(f=new_gelu),
                c_proj=nn.Linear(4 * embed_dim, embed_dim, bias=bias),
                dropout=nn.Dropout(dropout),
            )
        )

    def forward(self, x):
        assert x.dtype == torch.float32, f"TransformerBlock input dtype is {x.dtype}, but expected torch.float32"
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """GPT-inspired architecture"""

    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.Sequential(
            # Do the same as above, but with an OrderedDict:
            OrderedDict(
                [
                    (
                        f"block{i}",
                        TransformerBlock(
                            n_heads=config.n_heads,
                            embed_dim=config.embed_dim,
                            dropout=config.dropout,
                            bias=config.bias,
                            is_causal=config.is_causal,
                        ),
                    )
                    for i in range(config.n_layers)
                ]
            )
        )
        self.unembedding = nn.Sequential(
                nn.LayerNorm(
                    normalized_shape=config.embed_dim,
                ),
                nn.Linear(
                    in_features=config.embed_dim,
                    out_features=config.output_vocab_size,
                ),
            )
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith("c_proj.weight"):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))


    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: LongTensor,  # [B, seq_len]
        targets=None,
    ) -> Tensor:
        device = idx.device
        batch_size, seq_len = idx.size()
        assert (
            seq_len <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

        # --- forward the GPT model itself
        # token embeddings of shape (batch_size, seq_len, embed_dim)
        tok_emb = self.token_embedding(idx)  
        assert tok_emb.dtype == torch.float32, f"tok_emb dtype is {tok_emb.dtype}, but expected torch.float32"
        # position embeddings of shape (1, seq_len, embed_dim)
        pos_emb = self.pos_embedding(pos)  
        x = self.dropout(tok_emb + pos_emb)
        x = self.h(x)

        # unembedding: transform back to predicted next tokens
        y = self.unembedding(x)   # B T C @ . C V -> B T V
        
        return y
    
    # NOTE:
    # during training,  we only care about y[:, :-1, :]...
    # during inference, we only care about y[:, -1:, :]...

    # TODO: Add this to main configuration logic
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx