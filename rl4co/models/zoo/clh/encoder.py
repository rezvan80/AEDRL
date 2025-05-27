import torch.nn as nn
from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import Normalization, SkipConnection
from rl4co.models.zoo.clh.attention import CMHA
import torch


class CMHALayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_station,
        feedforward_hidden=512,
        normalization="batch",
    ):
        super(CMHALayer, self).__init__(
            SkipConnection(CMHA(num_heads, embed_dim, num_station, embed_dim)),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feedforward_hidden),
                    nn.GELU(),
                    nn.Linear(feedforward_hidden, embed_dim),
                )
                if feedforward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class CLHEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn=None,
        moe_kwargs: dict = None,
        num_station=None,
    ):
        super(CLHEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        # Map input to embedding space
        if init_embedding is None:
            self.init_embedding = env_init_embedding(env_name, {"embed_dim": embed_dim})
        else:
            self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                CMHALayer(
                    num_heads,
                    embed_dim,
                    num_station,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x, mask=None):
        init_embeds = self.init_embedding(x)
        if self.layers._modules["0"]._modules["0"].module.num_station != x[
            "stations"
        ].size(-2):
            for i in range(len(self.layers._modules)):
                self.layers._modules[str(i)]._modules["0"].module.num_station = x[
                    "stations"
                ].size(-2)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds
