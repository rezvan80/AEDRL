import os
import re
import numpy as np
import torch
from tensordict.tensordict import TensorDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))


def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


def load_evrp_to_tensordict(filename):
    f = open(filename, "r")
    content = f.read()
    # vehicles = torch.Tensor(
    #     [int(re.search("VEHICLES: (\d+)", content, re.MULTILINE).group(1))]
    # ).unsqueeze(0)
    # optimalValue = float(
    #     re.search("OPTIMAL_VALUE: (\d+)", content, re.MULTILINE).group(1)
    # )
    capacity = float(re.search("CAPACITY: (\d+)", content, re.MULTILINE).group(1))
    # dimension = int(re.search("DIMENSION: (\d+)", content, re.MULTILINE).group(1))
    station_number = int(re.search("STATIONS: (\d+)", content, re.MULTILINE).group(1))
    energy_capacity = float(
        re.search("ENERGY_CAPACITY: (\d+)", content, re.MULTILINE).group(1)
    )
    energy_consumption = float(
        re.search("ENERGY_CONSUMPTION: (\d+\.?\d*)", content, re.MULTILINE).group(1)
    )
    max_length=energy_capacity / energy_consumption/3
    demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    demand = torch.Tensor([float(b) for a, b in demand][1:]).unsqueeze(0)
    nodes = re.findall(r"^(\d+)( +)([+-]?\d+(?:\.\d+)?)( +)([+-]?\d+(?:\.\d+)?)", content, re.MULTILINE)
    nodes = torch.Tensor([[float(c), float(e)] for a, b, c,d,e in nodes])
    bias = torch.min(nodes, dim=0).values.unsqueeze(0)
    nodes = (nodes - bias) / max_length
    depot = nodes[0].unsqueeze(0)
    stations = nodes[-station_number:].unsqueeze(0)
    locs = nodes[1:-station_number].unsqueeze(0)
    # min_times = torch.full((1, nodes.size(-2)), 0.0)
    # max_times = torch.full((1, nodes.size(-2)), 3000)
    # time_windows = torch.stack((min_times, max_times), dim=-1)
    # durations = torch.full((1, nodes.size(-2)), 0)
    td = TensorDict(
        {
            "locs": locs,
            "depot": depot,
            "stations": stations,
            "demand": demand / capacity,
            "factor": torch.Tensor([max_length]),
            # "durations": durations,
            # "time_windows": time_windows,
        },
        batch_size=[1],
    )
    return td


def load_txt_to_tensordict(filename):
    f = open(filename, "r")
    content = f.read()
    capacity = float(
        re.search("C Vehicle load capacity /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    energy_capacity = float(
        re.search(
            "Q Vehicle fuel tank capacity /(\d+\.?\d*)/", content, re.MULTILINE
        ).group(1)
    )
    energy_consumption = float(
        re.search("r fuel consumption rate /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    velocity = float(
        re.search("v average Velocity /(\d+\.?\d*)/", content, re.MULTILINE).group(1)
    )
    recharge = float(
        re.search("g inverse refueling rate /(\d+\.?\d*)/", content, re.MULTILINE).group(
            1
        )
    )
    depot_add = re.findall(
        r"d          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)       (\d+\.?\d* ?)       (\d+\.?\d*[ ]{0,3})     (\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    depot_add = torch.Tensor(
        [[float(a), float(b), float(e)] for a, b, c, d, e in depot_add]
    )
    stations = re.findall(
        r"f          (-?\d+\.?\d* ?)       (-?\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    stations = torch.Tensor([[float(a), float(b)] for a, b in stations])
    customs = re.findall(
        r"c          (-?\d+\.?\d* ?)       (-?\d+\.?\d* ?)       (\d+\.?\d* ?)       (\d+\.?\d*[ ]{0,3})     (\d+\.?\d*[ ]{0,2})     (\d+\.?\d*)",
        content,
        re.MULTILINE,
    )
    customs = torch.Tensor(
        [
            [float(a), float(b), float(c), float(d), float(e), float(f)]
            for a, b, c, d, e, f in customs
        ]
    )
    demand = (customs[:, 2] / capacity).unsqueeze(0)
    customs_loc = customs[:, :2]
    max_length = energy_capacity / energy_consumption / velocity
    charge_time = energy_capacity / recharge / max_length
    customs_start = customs[:, 3] / max_length
    customs_end = customs[:, 4] / max_length
    service = customs[:, 5] / max_length
    max_time = depot_add[0, 2] / max_length
    depot = depot_add[:, :2]
    nodes = torch.cat((depot, stations, customs_loc), dim=-2)
    bias = torch.min(nodes, dim=0).values.unsqueeze(0)
    depot = ((depot - bias) / max_length).cuda()
    stations = ((stations - bias) / max_length).unsqueeze(0).cuda()
    customs = ((customs_loc - bias) / max_length).unsqueeze(0).cuda()
    duration_other = torch.full((1, 1 + stations.size(-2)), charge_time).cuda()
    durations = torch.cat((duration_other, service.unsqueeze(0).cuda()), dim=-1)
    min_times = torch.full((1, 1 + stations.size(-2) + customs.size(-2)), 0.0).cuda()
    max_times = torch.full((1, 1 + stations.size(-2) + customs.size(-2)), max_time).cuda()
    min_times[:, 1 + stations.size(-2) :] = customs_start.unsqueeze(0).cuda()
    max_times[:, 1 + stations.size(-2) :] = customs_end.unsqueeze(0).cuda()
    time_windows = torch.stack((min_times, max_times), dim=-1)
    td = TensorDict(
        {
            "locs": customs,
            "depot": depot,
            "stations": stations,
            "demand": demand,
            "factor": torch.Tensor([max_length]),
            "durations": durations,
            "time_windows": time_windows,
        },
        batch_size=[1],
    )
    return td


def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def load_solomon_instance(name, path=None, edge_weights=False):
    """Load solomon instance from a file"""
    import vrplib

    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.txt"
    if not os.path.isfile(file_path):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(
        path=file_path,
        instance_format="solomon",
        compute_edge_weights=edge_weights,
    )


def load_solomon_solution(name, path=None):
    """Load solomon solution from a file"""
    import vrplib

    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.sol"
    if not os.path.isfile(file_path):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=file_path)
