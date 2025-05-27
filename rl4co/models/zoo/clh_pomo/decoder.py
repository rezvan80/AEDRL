from dataclasses import dataclass, fields
from typing import Tuple, Union
import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor
import torch.nn.functional as F
from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import (
    env_context_embedding,
)
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


class CLHDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        # dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )

        if pointer is None:
            # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
            pointer_attn_class = (
                PointerAttention if moe_kwargs is None else PointerAttnMoE
            )
            pointer = pointer_attn_class(
                embed_dim,
                num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )

        self.pointer = pointer

        self.project_node_embeddings_one = nn.Linear(
            embed_dim, 2 * embed_dim, bias=linear_bias
        )
        # self.project_problem_size_kv = nn.Sequential(
        #     nn.Linear(embed_dim, 2 * embed_dim, bias=linear_bias),
        #     nn.Linear(2 * embed_dim, embed_dim, bias=linear_bias),
        # )
        self.project_node_embeddings_two = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.project_set_context = nn.Linear(embed_dim * 3, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context
        self.FF_d = nn.Sequential(
            nn.Linear(2, 16, bias=linear_bias),
            nn.GELU(),
            nn.Linear(16, 1, bias=linear_bias),
        )
        # self.FF_solution=nn.Sequential(
        #     nn.Linear(2*embed_dim+2, 8*embed_dim, bias=linear_bias),
        #     nn.GELU(),
        #     nn.Linear(8*embed_dim, embed_dim, bias=linear_bias),
        # )
        self.local_query = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.local_key = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.history_query = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.history_key = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.solution_pool_val = None
        self.reward_val = None
        self.local=None
        self.history=None

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings

        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)
        context = self.context_embedding(node_embeds_cache, td)
        if (td["locs"].size(-2)==386)|(td["locs"].size(-2)==925):
            batch_index = torch.arange(
                td.batch_size[0],
                dtype=torch.int32,
                device=node_embeds_cache.device,
            )[:, None].expand_as(td["current_node"])
            step_context = self.project_set_context(
                torch.cat(
                    (
                        context,
                        self.local[batch_index, td["current_node"]].squeeze(-2),
                        self.history[batch_index, td["current_node"]].squeeze(-2),
                    ),
                    dim=-1,
                )
            )
        else:
            batch_index = torch.arange(
                td.batch_size[0],
                dtype=torch.int32,
                device=node_embeds_cache.device,
            )[:, None].expand_as(td["current_node"].squeeze(-1))
            step_context = self.project_set_context(
                torch.cat(
                    (
                        context,
                        self.local[batch_index, td["current_node"].squeeze(-1)],
                        self.history[batch_index, td["current_node"].squeeze(-1)],
                    ),
                    dim=-1,
                )
            )
        # step_context=context
        glimpse_q = graph_context_cache + step_context
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:

        if num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        # Compute logits
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        if num_starts > 1:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        (glimpse_key_fixed, glimpse_val_fixed) = self.project_node_embeddings_one(
            embeddings
            # + self.project_problem_size_kv(
            #     position_encoding_init(
            #         embeddings.size(0),
            #         embeddings.size(1),
            #         embeddings.size(2),
            #         embeddings.device,
            #     )
            # )
        ).chunk(2, dim=-1)
        logit_key_fixed = self.project_node_embeddings_two(embeddings)
        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

    def scaled_dot_product_attention(self, q, k, v, distance,reward=None):
        score = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        scores = self.FF_d(
            torch.cat(
                (score[:, :, :, None], distance[:, :, :, None]),
                dim=-1,
            )
        ).squeeze(-1)
        if reward is not None:
            scores=scores/reward
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)


def position_encoding_init(batch_size, n_position, emb_dim, device):

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.zeros(n_position, emb_dim).to(device)
    position_enc[1:] = (
        torch.arange(1, n_position, device=device)[:, None].expand(
            n_position - 1, emb_dim
        )
        / torch.pow(
            torch.ones(n_position - 1, emb_dim, device=device) * 10000,
            torch.arange(emb_dim, device=device)[None, :].expand(n_position - 1, emb_dim)
            // 2
            * 2
            / emb_dim,
        )
    ).to(device)
    position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1
    position_encoding = torch.mean(position_enc, dim=0)
    return position_encoding[None, None, :].expand(batch_size, n_position, emb_dim)
