import argparse
import logging
import os
import sys
from rl4co.utils.ops import get_distance
from typing import List, Union

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


DISTRIBUTIONS_PER_PROBLEM = {
    "tsp": [None],
    "vrp": [None],
    "pctsp": [None],
    "op": ["const", "unif", "dist"],
    "mdpp": [None],
    "pdp": [None],
    "evrp": [None],
    "evrptw": [None],
}


def generate_env_data(env_type, *args, **kwargs):
    """Generate data for a given environment type in the form of a dictionary"""
    try:
        # breakpoint()
        # remove all None values from args
        args = [arg for arg in args if arg is not None]

        return getattr(sys.modules[__name__], f"generate_{env_type}_data")(
            *args, **kwargs
        )
    except AttributeError:
        raise NotImplementedError(f"Environment type {env_type} not implemented")


def generate_tsp_data(dataset_size, tsp_size):
    return {
        "locs": np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)
    }


def generate_vrp_data(dataset_size, vrp_size, capacities=None):
    # From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
    CAPACITIES = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v
    return {
        "depot": np.random.uniform(size=(dataset_size, 2)).astype(
            np.float32
        ),  # Depot location
        "locs": np.random.uniform(size=(dataset_size, vrp_size, 2)).astype(
            np.float32
        ),  # Node locations
        "demand": np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
            np.float32
        ),  # Demand, uniform integer 1 ... 9
        "capacity": np.full(dataset_size, CAPACITIES[vrp_size]).astype(np.float32),
    }  # Capacity, same for whole dataset


def generate_evrp_data(dataset_size, num_size, stations, capacities=None):
    # From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    x1 = np.random.uniform(0, 0.5, (dataset_size, stations, 1))
    x2 = np.random.uniform(0.5, 1, (dataset_size, stations, 1))
    y1 = np.random.uniform(0, 0.5, (dataset_size, stations, 1))
    y2 = np.random.uniform(0.5, 1, (dataset_size, stations, 1))
    station = np.zeros((dataset_size, stations, 2))
    for i in range(stations):
        if i % 4 == 0:
            station[:, i] = np.concatenate((x1[:, i], y1[:, i]), axis=-1)
        elif i % 4 == 1:
            station[:, i] = np.concatenate((x1[:, i], y2[:, i]), axis=-1)
        elif i % 4 == 2:
            station[:, i] = np.concatenate((x2[:, i], y1[:, i]), axis=-1)
        else:
            station[:, i] = np.concatenate((x2[:, i], y2[:, i]), axis=-1)
    return {
        "depot": (np.random.uniform(0, 1, (dataset_size, 2))).astype(
            np.float32
        ),  # Depot location
        "locs": np.random.uniform(0, 1, size=(dataset_size, num_size, 2)).astype(
            np.float32
        ),  # Node locations
        "stations": station.astype(np.float32),
        "demand": (np.random.uniform(0, 1, size=(dataset_size, num_size))).astype(
            np.float32
        ),
    }


def generate_evrptw_data(dataset_size, num_size, stations, capacities=None):
    max_time = 24
    charge_time = 1.5
    x1 = np.random.uniform(0, 0.5, (dataset_size, stations, 1))
    x2 = np.random.uniform(0.5, 1, (dataset_size, stations, 1))
    y1 = np.random.uniform(0, 0.5, (dataset_size, stations, 1))
    y2 = np.random.uniform(0.5, 1, (dataset_size, stations, 1))
    station = np.zeros((dataset_size, stations, 2))
    for i in range(stations):
        if i % 4 == 0:
            station[:, i] = np.concatenate((x1[:, i], y1[:, i]), axis=-1)
        elif i % 4 == 1:
            station[:, i] = np.concatenate((x1[:, i], y2[:, i]), axis=-1)
        elif i % 4 == 2:
            station[:, i] = np.concatenate((x2[:, i], y1[:, i]), axis=-1)
        else:
            station[:, i] = np.concatenate((x2[:, i], y2[:, i]), axis=-1)
    depot = np.random.uniform(0, 0.6, (dataset_size, 2))
    locs = np.random.uniform(0, 1, size=(dataset_size, num_size, 2))
    durations_custom = np.random.uniform(size=(dataset_size, num_size))
    durations_other = np.full((dataset_size, stations + 1), charge_time)
    durations = np.concatenate((durations_other, durations_custom), axis=-1)
    temp = np.concatenate((station, locs), axis=-2)
    dist = np.sqrt(np.sum((np.expand_dims(depot,1) - temp) ** 2, axis=-1))
    dist = np.concatenate((np.zeros(shape=(dataset_size, 1)), dist), axis=1)
    upper_bound = max_time - dist - durations

    # 3. create random values between 0 and 1
    ts_1 = np.random.uniform(size=(dataset_size, num_size + 1 + stations))
    ts_2 = np.random.uniform(size=(dataset_size, num_size + 1 + stations))

    # 4. scale values to lie between their respective min_time and max_time and convert to integer values
    min_ts = dist + (upper_bound - dist) * ts_1
    max_ts = dist + (upper_bound - dist) * ts_2

    # 5. set the lower value to min, the higher to max
    min_times = np.minimum(min_ts, max_ts)
    max_times = np.maximum(min_ts, max_ts)

    # 6. reset times for depot
    min_times[..., :, : 1 + stations] = 0.0
    max_times[..., :, 0] = max_time
    max_times[..., :, 1 : 1 + stations] = max_time - dist[..., :, 1 : 1 + stations]

    # 7. ensure min_times < max_times to prevent numerical errors in attention.py
    # min_times == max_times may lead to nan values in _inner_mha()
    mask = min_times == max_times
    if np.any(mask):
        min_tmp = min_times.copy()
        min_tmp[mask] = np.maximum(
            dist[mask], min_tmp[mask] - 1
        )  # we are handling integer values, so we can simply substract 1
        min_times = min_tmp

        mask = min_times == max_times  # update mask to new min_times
        if np.any(mask):
            max_tmp = max_times.clone()
            max_tmp[mask] = np.minimum(
                np.floor(upper_bound[mask]),
                np.maximum(
                    np.ceil(min_tmp[mask] + durations[mask]),
                    max_tmp[mask] + 1,
                ),
            )
            max_times = max_tmp

    # 8. stack to tensor time_windows
    time_windows = np.stack((min_times, max_times), axis=-1)
    return {
        "depot": depot.astype(np.float32),  # Depot location
        "locs": locs.astype(np.float32),  # Node locations
        "stations": station.astype(np.float32),
        "demand": (np.random.uniform(0, 0.25, size=(dataset_size, num_size))).astype(
            np.float32
        ),
        "durations": durations.astype(np.float32),
        "time_windows": time_windows.astype(np.float32),
    }


def generate_pdp_data(dataset_size, pdp_size):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pdp_size, 2))
    return {
        "locs": loc.astype(np.float32),
        "depot": depot.astype(np.float32),
    }


def generate_op_data(dataset_size, op_size, prize_type="const", max_lengths=None):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == "const":
        prize = np.ones((dataset_size, op_size))
    elif prize_type == "unif":
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.0
    else:  # Based on distance to depot
        assert prize_type == "dist"
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (
            1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)
        ) / 100.0

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
    max_lengths = MAX_LENGTHS if max_lengths is None else max_lengths

    return {
        "depot": depot.astype(np.float32),
        "locs": loc.astype(np.float32),
        "prize": prize.astype(np.float32),
        "max_length": np.full(dataset_size, max_lengths[op_size]).astype(np.float32),
    }


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3, max_lengths=None):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
    max_lengths = MAX_LENGTHS if max_lengths is None else max_lengths
    penalty_max = max_lengths[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = (
        np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)
    )

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = (
        np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2
    )

    return {
        "locs": loc.astype(np.float32),
        "depot": depot.astype(np.float32),
        "penalty": penalty.astype(np.float32),
        "deterministic_prize": deterministic_prize.astype(np.float32),
        "stochastic_prize": stochastic_prize.astype(np.float32),
    }


def generate_mdpp_data(
    dataset_size,
    size=10,
    num_probes_min=2,
    num_probes_max=5,
    num_keepout_min=1,
    num_keepout_max=50,
    lock_size=True,
):
    """Generate data for the nDPP problem.
    If `lock_size` is True, then the size if fixed and we skip the `size` argument if it is not 10.
    This is because the RL environment is based on a real-world PCB (parametrized with data)
    """
    if lock_size and size != 10:
        # log.info("Locking size to 10, skipping generate_mdpp_data with size {}".format(size))
        return None

    bs = dataset_size  # bs = batch_size to generate data in batch
    m = n = size
    if isinstance(bs, int):
        bs = [bs]

    locs = np.stack(np.meshgrid(np.arange(m), np.arange(n)), axis=-1).reshape(-1, 2)
    locs = locs / np.array([m, n], dtype=np.float32)
    locs = np.expand_dims(locs, axis=0)
    locs = np.repeat(locs, bs[0], axis=0)

    available = np.ones((bs[0], m * n), dtype=bool)

    probe = np.random.randint(0, high=m * n, size=(bs[0], 1))
    np.put_along_axis(available, probe, False, axis=1)

    num_probe = np.random.randint(num_probes_min, num_probes_max + 1, size=(bs[0], 1))
    probes = np.zeros((bs[0], m * n), dtype=bool)
    for i in range(bs[0]):
        p = np.random.choice(m * n, num_probe[i], replace=False)
        np.put_along_axis(available[i], p, False, axis=0)
        np.put_along_axis(probes[i], p, True, axis=0)

    num_keepout = np.random.randint(num_keepout_min, num_keepout_max + 1, size=(bs[0], 1))
    for i in range(bs[0]):
        k = np.random.choice(m * n, num_keepout[i], replace=False)
        np.put_along_axis(available[i], k, False, axis=0)

    return {
        "locs": locs.astype(np.float32),
        "probe": probes.astype(bool),
        "action_mask": available.astype(bool),
    }


def generate_dataset(
    filename: Union[str, List[str]] = None,
    data_dir: str = "data",
    name: str = None,
    problem: Union[str, List[str]] = "all",
    data_distribution: str = "all",
    dataset_size: int = 10000,
    graph_sizes: Union[int, List[int]] = [21, 22, 29, 32, 50, 75, 100],
    station_sizes: Union[int, List[int]] = [7, 7, 5, 4, 9, 13, 11],
    overwrite: bool = False,
    seed: int = 1234,
    disable_warning: bool = True,
    distributions_per_problem: Union[int, dict] = None,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict

    Args:
        filename: Filename to save the data to. If None, the data is saved to data_dir/problem/problem_graph_size_seed.npz. Defaults to None.
        data_dir: Directory to save the data to. Defaults to "data".
        name: Name of the dataset. Defaults to None.
        problem: Problem to generate data for. Defaults to "all".
        data_distribution: Data distribution to generate data for. Defaults to "all".
        dataset_size: Number of datasets to generate. Defaults to 10000.
        graph_sizes: Graph size to generate data for. Defaults to [20, 50, 100].
        overwrite: Whether to overwrite existing files. Defaults to False.
        seed: Random seed. Defaults to 1234.
        disable_warning: Whether to disable warnings. Defaults to True.
        distributions_per_problem: Number of distributions to generate per problem. Defaults to None.
    """

    if isinstance(problem, list) and len(problem) == 1:
        problem = problem[0]

    graph_sizes = [graph_sizes] if isinstance(graph_sizes, int) else graph_sizes
    station_sizes = [station_sizes] if isinstance(station_sizes, int) else station_sizes
    if distributions_per_problem is None:
        distributions_per_problem = DISTRIBUTIONS_PER_PROBLEM
    if problem == "all":
        problems = distributions_per_problem
    else:
        problems = {
            problem: (
                distributions_per_problem[problem]
                if data_distribution == "all"
                else [data_distribution]
            )
        }

    # Support multiple filenames if necessary
    filenames = [filename] if isinstance(filename, str) else filename
    iter = 0

    # Main loop for data generation. We loop over all problems, distributions and sizes
    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for index in range(len(graph_sizes)):
                if filename is None:
                    datadir = os.path.join(data_dir, problem)
                    os.makedirs(datadir, exist_ok=True)
                    fname = os.path.join(
                        datadir,
                        "{}{}_{}{}_{}_seed{}.npz".format(
                            problem,
                            (
                                "_{}".format(distribution)
                                if distribution is not None
                                else ""
                            ),
                            graph_sizes[index],
                            (
                                "_{}".format(station_sizes[index])
                                if station_sizes[index] is not None
                                else ""
                            ),
                            name,
                            seed,
                        ),
                    )
                else:
                    try:
                        fname = filenames[iter]
                        # make directory if necessary
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        iter += 1
                    except Exception:
                        raise ValueError(
                            "Number of filenames does not match number of problems"
                        )
                    fname = check_extension(filename, extension=".npz")

                if not overwrite and os.path.isfile(
                    check_extension(fname, extension=".npz")
                ):
                    if not disable_warning:
                        log.info(
                            "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                                fname
                            )
                        )
                    continue

                # Set seed
                np.random.seed(seed)

                # Automatically generate dataset
                dataset = generate_env_data(
                    problem, dataset_size, graph_sizes[index], station_sizes[index]
                )
                # A function can return None in case of an error or a skip
                if dataset is not None:
                    # Save to disk as dict
                    log.info("Saving {} dataset to {}".format(problem, fname))
                    np.savez(fname, **dataset)


def generate_default_datasets(data_dir, generate_eda=False):
    """Generate the default datasets used in the paper and save them to data_dir/problem"""
    generate_dataset(data_dir=data_dir, name="val", problem="evrp", seed=4321)

    # By default, we skip the EDA datasets since they can easily be generated on the fly when needed
    if generate_eda:
        generate_dataset(
            data_dir=data_dir,
            name="test",
            problem="mdpp",
            seed=1234,
            graph_sizes=[10],
            dataset_size=100,
        )  # EDA (mDPP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", help="Filename of the dataset to create (ignores datadir)"
    )
    parser.add_argument(
        "--data_dir",
        default="data/evrp",
        help="Create datasets in data_dir/problem (default 'data')",
    )
    parser.add_argument(
        "--name", type=str, default="rand", help="Name to identify dataset"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="evrp",
        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
        " or 'all' to generate all",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="all",
        help="Distributions to generate for problem, default 'all'.",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=320, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes",
        type=int,
        default=100,
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument(
        "--station_sizes",
        type=int,
        default=8,
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument(
        "-f", action="store_true", help="Set true to overwrite", default=True
    )
    parser.add_argument("--seed", type=int, default=4321, help="Random seed")
    parser.add_argument("--disable_warning", action="store_true", help="Disable warning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.overwrite = args.f
    delattr(args, "f")
    generate_dataset(**vars(args))
