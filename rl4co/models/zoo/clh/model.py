from typing import Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.clh.policy import CLHPolicy


class CLH(REINFORCE):

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: CLHPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = CLHPolicy(
                env_name=env.name,
                env_num_station=env.generator.num_station,
                **policy_kwargs,
            )

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
