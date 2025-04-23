import argparse
import os
import json
import statistics
import re

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default="./output/dtu/")
parser.add_argument('--out_json', type=str, default="./output/dtu/out_json.json")
parser.add_argument('--iter', type=int, default=6000)
parser.add_argument('--config', type=str, default=None)

args = parser.parse_args()
path = args.path
iter = args.iter
out_json = args.out_json
config = args.config


# {
#     "ours_6000": {
#     "SSIM": 0.6313242316246033,
#     "SSIM_sk": 0.6100044846534729,
#     "PSNR": 17.61285400390625,
#     "LPIPS": 0.24291515350341797
#     }
# }
import numpy as np 

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

def compute_avg_error(psnr, ssim, lpips):
  """The 'average' error used in the paper."""
  mse = psnr_to_mse(psnr)
  dssim = np.sqrt(1 - ssim)
  return np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))


# 使用正则表达式来匹配不想要处理的文件名模式
exclude_pattern = re.compile(r'_old(_psnr\d+(\.\d+)?)?$', re.IGNORECASE)

names = ["SSIM", "SSIM_sk", "PSNR", "LPIPS"]
# names = ["PSNR"]
metrics = [[] for _ in range(len(names))]
 
for dir in os.listdir(path):
    dir_path = os.path.join(path, dir)
    if os.path.isdir(dir_path) and not exclude_pattern.search(dir):
        print(f'dir:{dir}')
        result = os.path.join(path, dir, "results_eval_mask.json")
        if os.path.exists(result):
            # print(result)
            with open(result, "r") as f:
                data = json.load(f)
            metric = data[f"ours_{iter}"]
            for i,name in enumerate(names):
                metrics[i].append(metric[name])
out = {}
if os.path.exists(out_json):
    with open(out_json, "r") as f:
        out = json.load(f)

count = len(metrics[0])
out[config] = {}
out[config]["count"] = count
for i,name in enumerate(names):
    # out[config][name] = metrics[i]
    if count > 1:
        mean = statistics.mean(metrics[i])
        variance = statistics.variance(metrics[i])
    else:
        mean = metrics[i][0]
        variance = 0
    out[config][f"{name}_mean"] = mean
    out[config][f"{name}_variance"] = variance

out[config]["AVG"] = compute_avg_error(out[config]["PSNR_mean"], out[config]["SSIM_sk_mean"], out[config]["LPIPS_mean"])

with open(out_json, "w") as f:
    json.dump(out, f)

