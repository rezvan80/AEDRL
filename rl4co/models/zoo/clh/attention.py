import math


import torch
import torch.nn as nn


class CMHA(nn.Module):
    def __init__(
        self,
        num_heads,
        input_dim,
        num_station,
        embed_dim=None,
        val_dim=None,
        key_dim=None,
    ):
        super(CMHA, self).__init__()
        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim
        self.num_station = num_station
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query_depot = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        self.W_query_station = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key_other = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val_other = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        self.W_query_custom = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key_all = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val_all = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))

        self.W_out = nn.Parameter(torch.Tensor(num_heads, key_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        batch_size, graph_size, input_dim = h.size()
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        num_task = graph_size - self.num_station - 1

        # depot->custom
        hflat = h[:, 1 + self.num_station :].contiguous().view(-1, input_dim)
        qflat = q[:, :1].contiguous().view(-1, input_dim)
        shp = (self.num_heads, batch_size, num_task, -1)
        shp_q = (self.num_heads, batch_size, 1, -1)
        Q = torch.matmul(qflat, self.W_query_depot).view(shp_q)
        K = torch.matmul(hflat, self.W_key_custom).view(shp)
        V = torch.matmul(hflat, self.W_val_custom).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)
        heads_depot = torch.matmul(attn, V)

        # station->other
        hflat = (
            torch.cat((h[:, :1], h[:, 1 + self.num_station :]), dim=1)
            .contiguous()
            .view(-1, input_dim)
        )
        qflat = q[:, 1 : 1 + self.num_station].contiguous().view(-1, input_dim)
        shp = (self.num_heads, batch_size, num_task + 1, -1)
        shp_q = (self.num_heads, batch_size, self.num_station, -1)
        Q = torch.matmul(qflat, self.W_query_station).view(shp_q)
        K = torch.matmul(hflat, self.W_key_other).view(shp)
        V = torch.matmul(hflat, self.W_val_other).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)
        heads_station = torch.matmul(attn, V)

        # targe->all
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q[:, 1 + self.num_station :].contiguous().view(-1, input_dim)
        shp = (self.num_heads, batch_size, graph_size, -1)
        shp_q = (self.num_heads, batch_size, num_task, -1)
        Q = torch.matmul(qflat, self.W_query_custom).view(shp_q)
        K = torch.matmul(hflat, self.W_key_all).view(shp)
        V = torch.matmul(hflat, self.W_val_all).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)
        heads_custom = torch.matmul(attn, V)

        heads = torch.cat((heads_depot, heads_station, heads_custom), dim=2)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, graph_size, self.embed_dim)
        # assert not torch.isnan(out).any()
        return out
