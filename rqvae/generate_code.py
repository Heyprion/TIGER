import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

DEFAULT_CONFIG = {
    "dataset": "Beauty",
    "ckpt_path": "./ckpt/Beauty/Jun-17-2025_15-21-52/best_collision_model.pth",
    "output_file": "../data/Beauty/Beauty_t5_rqvae.npy",
    "device": "cuda:0",
    "batch_size": 64,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate discrete codes from a trained RQVAE model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON/YAML config file.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the RQVAE checkpoint.")
    parser.add_argument("--output_file", type=str, default=None, help="Output .npy file path.")
    parser.add_argument("--device", type=str, default=None, help="Device, such as cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for code generation.")
    return parser.parse_args()


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files.") from exc
        with config_path.open("r") as f:
            loaded = yaml.safe_load(f)
    else:
        with config_path.open("r") as f:
            loaded = json.load(f)

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a dictionary/object at the top level.")
    return loaded


def build_runtime_config(cli_args):
    config = dict(DEFAULT_CONFIG)
    if cli_args.config:
        config.update(load_config(cli_args.config))

    for key in ["dataset", "ckpt_path", "output_file", "device", "batch_size"]:
        value = getattr(cli_args, key)
        if value is not None:
            config[key] = value

    if not config.get("output_file"):
        dataset = config["dataset"]
        config["output_file"] = f"../data/{dataset}/{dataset}_t5_rqvae.npy"

    return config


def main():
    cli_args = parse_args()
    runtime_config = build_runtime_config(cli_args)

    dataset = runtime_config["dataset"]
    ckpt_path = runtime_config["ckpt_path"]
    output_file = runtime_config["output_file"]
    device = torch.device(runtime_config["device"])
    batch_size = runtime_config["batch_size"]

    print("Runtime config:", runtime_config)

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args.data_path)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    tt = 0
    # There are often duplicate items in the dataset, and we no longer differentiate them.
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)

            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for _, value in all_indices_dict.items():
        code = [int(item.split("_")[1].strip(">")) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]

    if len(duplicates) > 0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving codes to {output_path}")
    print(f"the first 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)


if __name__ == "__main__":
    main()
