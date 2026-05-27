import argparse
import yaml
import os
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(flat_dict, sep='.'):
    unflat = {}
    for k, v in flat_dict.items():
        parts = k.split(sep)
        d = unflat
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return unflat

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def infer_type(value):
    if isinstance(value, bool):
        return str2bool
    return type(value) if value is not None else str

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args, _ = parser.parse_known_args()

    base_config = load_config(args.config)
    flat_base_config = flatten_dict(base_config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    for key, val in flat_base_config.items():
        if key == "data.other_regions":
            parser.add_argument(f"--{key}", type=str, default=",".join(val))  # handle as comma-separated string
        else:
            parser.add_argument(f"--{key}", type=infer_type(val), default=val)

    args = parser.parse_args()

    flat_args = {}
    for k, v in vars(args).items():
        if k == "data.other_regions":
            flat_args[k] = v.split(',') if isinstance(v, str) else v
        elif k != "config":
            flat_args[k] = v

    final_args = unflatten_dict(flat_args)

    if final_args['acquisition']['acquisition_function']['name'] == 'al':
        final_args['log']['log_dir'] = os.path.join(
            final_args['log']['log_dir'],
            f"{final_args['data']['task']}_{final_args['data']['src_region']}_{final_args['data']['tgt_region']}_"
            f"{final_args['acquisition']['acquisition_function']['name']}_"
            f"{final_args['acquisition']['acquisition_function']['method']}_{final_args['seed']['left']}_{final_args['seed']['right']}"
        )
    elif final_args['acquisition']['acquisition_function']['name'] == 'pda':
        final_args['log']['log_dir'] = os.path.join(
            final_args['log']['log_dir'],
            f"{final_args['data']['task']}_{final_args['data']['src_region']}_{final_args['data']['tgt_region']}_"
            f"{final_args['acquisition']['acquisition_function']['name']}_"
            f"{final_args['acquisition']['acquisition_function']['reward']}_{final_args['seed']['left']}_{final_args['seed']['right']}"
        )
    else:
        final_args['log']['log_dir'] = os.path.join(
            final_args['log']['log_dir'],
            f"{final_args['data']['task']}_{final_args['data']['src_region']}_{final_args['data']['tgt_region']}_"
            f"{final_args['acquisition']['acquisition_function']['name']}_{final_args['seed']['left']}_{final_args['seed']['right']}"
        )

    return final_args