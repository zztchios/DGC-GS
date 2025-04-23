import os
import configparser
import json
import statistics
from argparse import ArgumentParser


def copy_configs(default_config, configs_path, cnt):
    for i in range(1, cnt+1):
        os.system(f"cp {default_config} {configs_path}/config_{i}.ini")

def creat_configs(default_config, params, configs_path):
    config = configparser.ConfigParser()

    config_index = 0
    for param, value in params.items():
        for v in value:
            config.read(default_config)
            config_index += 1
            config.set("default", param, str(v))
            # print(config)
            config.write(open(os.path.join(configs_path, f"config-{param}-{v}.ini"), "w+"))

def run_experiments(args, models_path, configs_path):
    print("Running experiments...")
    for config_name in os.listdir(configs_path):
        print(f"Running experiments with config {config_name}")
        config_path = os.path.join(configs_path, config_name)
        for scan_id in args.scan_ids:
            model_path = os.path.join(models_path, config_name.split(".")[0], f"scan{scan_id}")
            os.makedirs(model_path, exist_ok=True)
            # os.system(f"cp {config_path} {model_path}/")
            for i in range(1,args.count+1):
                print(f"Running experiments with config {config_name} {i}")
                os.system(f"bash scripts/run_dtu.sh {args.dtu_path}/scan{scan_id} {model_path}/{i} {args.device} scan{scan_id} {config_path} {args.iter}")
        print(f"Run experiments with config {config_name} DONE")

def get_results(args, models_path):
    print(f"Getting results...")
    result_file = os.path.join(models_path, "result.json")
    for config_dir in os.listdir(models_path):
        if config_dir == "result.json":
            continue
        get_config_results(args, os.path.join(models_path, config_dir), result_file)
    print(f"Get results DONE, written to {result_file}")


def get_config_results(args, config_path, result_file):
    config_dir = os.path.basename(config_path)
    print(f"Getting config {config_dir} results...")

    metrics = [[[f"scan{scan_id}"] for scan_id in args.scan_ids] for _ in range(len(args.names))]
    metrics = {}
    for name in args.names:
        metrics[name] = {}
        for scan_id in args.scan_ids:
            metrics[name][f"scan{scan_id}"] = []

    for i, scan_dir in enumerate(os.listdir(config_path)):
        for count_dir in os.listdir(os.path.join(config_path, scan_dir)):
            result = os.path.join(config_path, scan_dir, count_dir, "results_eval_mask.json")
            if os.path.exists(result):
                print(result)
                with open(result, "r") as f:
                    data = json.load(f)
                metric = data[f"ours_{args.iter}"]
                for j,name in enumerate(args.names):
                    metrics[name][scan_dir].append(metric[name])
    print(metrics)
    out = {}
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            out = json.load(f)

    out[config_dir] = {f"scan{scan_id}":{} for scan_id in args.scan_ids}
    out[config_dir]["all_scan"] = {}
    # for scan_metric in metrics[0]:
    #     scan = scan_metric[0]
    #     out[config_dir][scan] = {}
    scan_max_idx = {}
    for scan, scan_metric in metrics["PSNR"].items():
        idx = -1
        if scan_metric:
            idx = scan_metric.index(max(scan_metric))
        scan_max_idx[scan] = idx

    scan_max_metric = {name: [] for name in args.names}
    for i,name in enumerate(args.names):
        mean = []
        for scan,scan_metric in metrics[name].items():
            mean += scan_metric
            scan_count = len(scan_metric)
            if scan_count > 0:
                if scan_count > 1:
                    scan_mean = statistics.mean(scan_metric)
                    scan_variance = statistics.variance(scan_metric)
                else:
                    scan_mean = scan_metric[0]
                    scan_variance = 0

                scan_max = scan_metric[scan_max_idx[scan]]
                out[config_dir][scan][f"{name}_mean"] = scan_mean
                out[config_dir][scan][f"{name}_variance"] = scan_variance
                out[config_dir][scan][f"count"] = scan_count
                out[config_dir][scan][f"{name}_max"] = scan_max
                scan_max_metric[name].append(scan_max)
        all_count = len(mean)
        if all_count > 0:
            if all_count > 1:
                all_max_mean = statistics.mean(scan_max_metric[name])
                all_mean = statistics.mean(mean)
                all_variance = statistics.variance(mean)
            else:
                all_max_mean = scan_max_metric[name][0]
                all_mean = mean[0]
                all_variance = 0
    
            out[config_dir]["all_scan"][f"{name}_max_mean"] = all_max_mean
            out[config_dir]["all_scan"][f"{name}_mean"] = all_mean
            out[config_dir]["all_scan"][f"{name}_variance"] = all_variance
            out[config_dir]["all_scan"][f"count"] = all_count
    
    with open(result_file, "w") as f:
        json.dump(out, f)
    print(f"Get config {config_dir} results DONE.")


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--scan_ids', nargs="+", type=int, default=[40, 8, 21, 30, 31, 34, 38, 40, 41, 45, 63, 82, 103, 110, 114])
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--dtu_path', type=str, default="data/dtu")
    parser.add_argument('--output_path', type=str, default="test")
    parser.add_argument('--names', nargs="+", type=str, default=["PSNR", "SSIM", "SSIM_sk", "LPIPS"])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--iter', type=int, default=6000)
    parser.add_argument('--configs_path', type=str, default="test/configs")

    args = parser.parse_args()

    models_path = os.path.join(args.output_path, "models")
    os.makedirs(models_path, exist_ok=True)
    configs_path = os.path.join(args.output_path, "configs")
    os.makedirs(configs_path, exist_ok=True)
    os.system(f"cp -a {args.configs_path}/* {configs_path}/")

    # run_experiments(args, models_path, configs_path)
    get_results(args, models_path)
