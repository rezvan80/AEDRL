from typing import Optional
import os
import torch

from tensordict.tensordict import TensorDict

from rl4co.data.utils import (
    load_txt_to_tensordict,
    load_npz_to_tensordict,
    load_evrp_to_tensordict,
)
from rl4co.envs.routing.evrp.env import EVRPEnv
from rl4co.utils.ops import gather_by_index
from ..evrp.generator import EVRPGenerator
from .generator import EVRPTWGenerator

from .render import render


class EVRPTWEnv(EVRPEnv):

    name = "evrptw"

    def __init__(
        self,
        generator: EVRPTWGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPTWGenerator(**generator_params)
        self.generator = generator

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td["locs"].shape[0]
        current_node = td["action"][:, None]
        batch_index = torch.arange(td.batch_size[0])[:, None].to(device=td.device)
        # update current_time
        distance = (
            td["locs"][batch_index, td["current_node"]]
            - td["locs"][batch_index, current_node]
        ).norm(p=2, dim=-1)
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        end_times = gather_by_index(td["time_windows"], td["action"])[..., 1].reshape(
            [batch_size, 1]
        )
        td["penalty_time"][td["current_time"] + distance > end_times] += (
            td["current_time"][td["current_time"] + distance > end_times]
            + distance[td["current_time"] + distance > end_times]
            - end_times[td["current_time"] + distance > end_times]
        )
        td["current_time"] = (current_node != 0).bool() * (
            torch.max(td["current_time"] + distance, start_times) + duration
        )
        # current_node is updated to the selected action
        n_loc = td["demand"].size(-1)
        demand = torch.nn.functional.pad(td["demand"], (1, 0), mode="constant", value=0)
        num_station = td["locs"].size(-2) - td["demand"].size(-1) - 1
        selected_demand = gather_by_index(
            demand,
            torch.clamp(current_node - num_station, 0, n_loc),
            squeeze=False,
        )
        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()
        used_length = (
            (
                td["locs"][torch.arange(td.batch_size[0],device=td.device)[:, None], td["current_node"]]
                - td["locs"][torch.arange(td.batch_size[0],device=td.device)[:, None], current_node]
            ).norm(p=2, dim=-1)
            + td["used_length"]
        ) * (current_node > num_station).float()
        visited = td["visited"].clone()
        visit = visited.scatter(-1, current_node, 1)
        visit[...,1:1+num_station][(current_node>num_station).expand(batch_size,num_station)]=0
        done = (visit[...,1+num_station:].sum(-1, keepdim=True) == visit[...,1+num_station:].size(-1)) & (current_node == 0)
        if done.any():
            reward = torch.zeros_like(done)
            td.update(
                {
                    "current_node": current_node,
                    "used_capacity": used_capacity,
                    "visited": visit,
                    "reward": reward,
                    "done": done,
                    "used_length": used_length,
                }
            )
        else:
            td.update(
                {
                    "current_node": current_node,
                    "used_capacity": used_capacity,
                    "visited": visit,
                    "done": done,
                    "used_length": used_length,
                }
            )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat(
                    (td["depot"][..., None, :], td["stations"], td["locs"]), -2
                ),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                "penalty_time": torch.zeros(
                    *batch_size,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                "used_length": torch.zeros((*batch_size, 1), device=device),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + td["stations"].size(-2) + 1),
                    dtype=torch.bool,
                    device=device,
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        if "factor" in td.sorted_keys:
            td_reset.set(
                "factor",
                td["factor"],
            )
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        exceeds_cap = td["demand"] + td["used_capacity"] > 1
        batch_index = torch.arange(td.batch_size[0],device=td.device)[:, None]
        num_station = td["locs"].size(-2) - td["demand"].size(-1) - 1
        length_loc = (
            (
                td["locs"][batch_index, td["current_node"]]
                - td["locs"][:, 1 + num_station :]
            ).norm(p=2, dim=-1)[:, :, None]
            + (
                td["locs"][:, 1 + num_station :, None]
                - td["locs"][:, None, : 1 + num_station]
            ).norm(p=2, dim=-1)
            + td["used_length"][:, :, None]
        )
        exceeds_length = (length_loc + 1e-4 > 3).all(-1)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][...,num_station+1:].to(exceeds_cap.dtype) | exceeds_cap | exceeds_length
        length_other = (
            td["locs"][batch_index, td["current_node"]] - td["locs"][:, : 1 + num_station]
        ).norm(p=2, dim=-1) + td["used_length"]
        exceeds_station = length_other[:, 1:] > 3
        mask_station = exceeds_station | td["visited"][...,1:num_station+1]
        # Cannot visit the depot if just visited and still unserved nodes
        exceeds_depot = length_other[:, :1] > 3
        mask_depot = (
            (td["current_node"] == 0) & ~(td["visited"].bool().all(-1, keepdim=True))
        ) | exceeds_depot
        return ~torch.cat((mask_depot, mask_station, mask_loc), -1)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        return super()._get_reward(td, actions) - td["penalty_time"].squeeze(-1)

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        if os.path.splitext(fpath)[1] == ".txt":
            td_load = load_txt_to_tensordict(fpath)
        elif os.path.splitext(fpath)[1] == ".npz":
            td_load = load_npz_to_tensordict(fpath)
        elif os.path.splitext(fpath)[1] == ".evrp":
            td_load = load_evrp_to_tensordict(fpath)
        return td_load

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
