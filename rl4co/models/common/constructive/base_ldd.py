import abc
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
import math
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import gather_by_index, calculate_entropy, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated hidden state, TensorDict, and environment
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicy(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. Decode the action given the hidden state and the environment state at the current step
        4. Update the environment state with the action. Repeat 3-4 until all sequences are done
        5. Obtain log likelihood, rewards etc.

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(ConstructivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder and decoder
        if encoder is None:
            log.warning("`None` was provided as encoder. Using `NoEncoder`.")
            encoder = NoEncoder()
        self.encoder = encoder
        self.decoder = decoder

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)
        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        if (self.__module__ == "rl4co.models.zoo.clh_pomo.policy") | (
            self.__module__ == "rl4co.models.zoo.clh.policy"
        ):
            q = self.decoder.local_query(hidden.node_embeddings)
            k = self.decoder.local_key(hidden.node_embeddings)
            local_part = torch.topk(
                (
                    td["locs"][: hidden.node_embeddings.size(0), None]
                    - td["locs"][: hidden.node_embeddings.size(0), :, None]
                ).norm(p=2, dim=-1),
                td["locs"].size(-2) - td["demand"].size(-1) - 1 + 3,
                -1,
                False,
                True,
            )
            batch_index = torch.arange(
                hidden.node_embeddings.size(0),
                dtype=torch.int32,
                device=td.device,
            )[:, None]
            self.decoder.local = torch.zeros_like(
                hidden.node_embeddings,
                device=hidden.node_embeddings.device,
                dtype=hidden.node_embeddings.dtype,
            )
            for i in range(1, td["locs"].size(-2)):
                self.decoder.local[:, i : i + 1] = (
                    self.decoder.scaled_dot_product_attention(
                        q[:, i : i + 1],
                        k[batch_index, local_part.indices[:, :, 1:][:, i]],
                        hidden.node_embeddings[
                            batch_index, local_part.indices[:, :, 1:][:, i]
                        ],
                        local_part.values[:, :, 1:][:, i : i + 1],
                    )
                )
            k = td["demand"].size(-1)

            if phase == "val":
                if self.decoder.solution_pool_val is None:
                    self.decoder.solution_pool_val, self.decoder.reward_val = (
                        get_solution(td.clone(), k)
                    )
                solution_pool = self.decoder.solution_pool_val.clone()
                reward = self.decoder.reward_val.clone()
            elif phase == "train":
                solution_pool, reward = get_solution(td.clone(), k)
            elif ((td["locs"].size(-2)==386)|(td["locs"].size(-2)==925))&(phase=="test"):
                solution_pool, reward = get_solution(td.clone(), k)
            else:
                solution_pool, reward = get_solution_mul(td.clone(), k)
            next = torch.cat((solution_pool[:, 1:], solution_pool[:, :1]), dim=-1)
            self.decoder.history = torch.zeros_like(
                hidden.node_embeddings,
                device=hidden.node_embeddings.device,
                dtype=hidden.node_embeddings.dtype,
            )
            for j in range(1, solution_pool.max() + 1):
                self.decoder.history[:, j : j + 1] = (
                    self.decoder.scaled_dot_product_attention(
                        self.decoder.history_query(hidden.node_embeddings[:, j : j + 1]),
                        self.decoder.history_key(
                            hidden.node_embeddings[:, next[solution_pool == j].view(-1)]
                        ),
                        hidden.node_embeddings[:, next[solution_pool == j].view(-1)],
                        (
                            td["locs"][
                                : hidden.node_embeddings.size(0),
                                j : j + 1,
                            ]
                            - td["locs"][
                                : hidden.node_embeddings.size(0),
                                next[solution_pool == j].view(-1),
                            ]
                        ).norm(p=2, dim=-1)[:, None, :],
                        -reward.view(1, -1, 1)
                        .expand(
                            hidden.node_embeddings.size(0),
                            solution_pool.size(0),
                            solution_pool.size(-1),
                        )
                        .contiguous()
                        .view(hidden.node_embeddings.size(0), -1)[
                            :,
                            None,
                            solution_pool.contiguous().view(-1) == j,
                        ],
                    )
                )
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if phase == "test":
            print(actions)
            # print(torch.softmax(logprobs,-1)[td["reward"].max(-1).indices].round(decimals=3))
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict


def get_solution(td, k):
    p_size = td["demand"].size(-1)
    station_size = td["locs"].size(-2) - td["demand"].size(-1) - 1
    batch_size = td.batch_size[0]
    data = td["locs"][:batch_size]
    max_length = 3.0
    batch_index = torch.arange(batch_size, device=data.device, dtype=torch.int32)[:, None]
    demand = torch.nn.functional.pad(
        td["demand"][:batch_size], (1, 0), mode="constant", value=0
    )
    candidates = torch.zeros(batch_size, p_size + 1, device=data.device).bool()
    solution = []
    selected_node = torch.zeros(
        size=(batch_size, 1),
        device=data.device,
        dtype=torch.int64,
    )
    current_load = torch.zeros(
        size=(batch_size, 1),
        device=data.device,
        dtype=td["demand"].dtype,
    )
    current_length = torch.zeros(
        size=(batch_size, 1),
        device=data.device,
        dtype=td["demand"].dtype,
    )
    visited = candidates[:, 1:].clone().to(torch.uint8)
    while (visited.all(-1) != 1).any() | (selected_node != 0).any():
        dists = (data[batch_index, selected_node] - data).norm(p=2, dim=-1)
        dists[:, 1 + station_size :][candidates[:, 1:]] = 1e5
        dists[:, 1 : station_size + 1][
            (selected_node < station_size + 1).expand(batch_size, station_size)
        ] = 1e5
        dists.scatter_(-1, selected_node, 1e5)
        dists[:, :1][visited.all(-1, keepdim=True).bool()] = 0
        selected_demand = gather_by_index(
            demand,
            torch.clamp(selected_node - station_size, 0, p_size),
            squeeze=False,
            dim=-1,
        )
        current_load += selected_demand
        dists[:, 1 + station_size :][current_load + td["demand"][:batch_size] > 1] = 1e5
        exceed_length_loc = (
            (
                ((data[batch_index, selected_node] - data[:, 1 + station_size :])
                .norm(p=2, dim=-1)* (1 + 0.5 * current_load))
                .unsqueeze(-1)
                + (
                    data[:, 1 + station_size :, None] - data[:, None, : 1 + station_size]
                ).norm(p=2, dim=-1)* (1 + 0.5 * (current_load + demand[:, 1:, None]))
                + current_length[:, :, None]
            )
            + 1e-4
            > max_length
        ).all(-1)
        exceed_length_other = (
            (data[batch_index, selected_node] - data[:, : 1 + station_size]).norm(
                p=2, dim=-1
            )* (1 + 0.5 * current_load)
            + current_length
        ) > max_length
        exceed_length = torch.cat((exceed_length_other, exceed_length_loc), dim=-1)
        dists[exceed_length] = 1e5
        next_selected_node = dists.min(-1, keepdim=True)[1]
        current_load[next_selected_node == 0] = 0
        current_length += (
            data[batch_index, selected_node] - data[batch_index, next_selected_node]
        ).norm(p=2, dim=-1)* (1 + 0.5 * current_load)
        current_length[next_selected_node < station_size + 1] = 0
        solution.append(next_selected_node)
        candidates.scatter_(
            -1, torch.clamp(next_selected_node - station_size, 0, p_size), 1
        )
        visited = candidates[:, 1:].clone().to(visited.dtype)
        selected_node = next_selected_node
    pi = torch.stack(solution, -1).squeeze(-2)
    reward = get_reward(td[:batch_size], pi).unsqueeze(-1)
    if pi.size(0) <= k:
        return pi, reward
    pi = torch.cat((pi, reward), -1).squeeze(0)
    index_p = torch.topk(pi[:, -1], min(k, pi.size(0)), largest=True, sorted=True).indices
    pi = pi[index_p]
    x = torch.nonzero(pi[:, :-1])[:, -1].max() + 2
    solution = pi[:, :x].long()
    reward = pi[:, -1]
    return solution, reward


def get_reward(td, actions):
    batch_size = td["locs"].shape[0]
    graph_size = td["locs"].size(-2)
    depot = td["locs"][..., :1, :]
    locs_ordered = torch.cat(
        [
            depot,
            gather_by_index(
                td["locs"],
                actions,
                dim=-2,
            ).reshape([batch_size, actions.size(-1), 2]),
        ],
        dim=-2,
    )
    return -get_tour_length(locs_ordered)



