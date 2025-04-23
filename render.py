#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time

import numpy as np
import matplotlib.cm as cm


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)
            
def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, near=0):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    if near > 0:
        mask_near = None
        for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
            mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near
            mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
        gaussians.prune_points_inference(mask_near)
    
    render_time_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
        torch.cuda.synchronize()
        start = time.time()
        render_pkg = render(view, gaussians, pipeline, background, inference=True)
        torch.cuda.synchronize()
        end = time.time()
        
        render_time_list.append((end-start)*1000)
        
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        depth = (render_pkg['depth'] - render_pkg['depth'].min()) / (render_pkg['depth'].max() - render_pkg['depth'].min()) + 1 * (1 - render_pkg["alpha"])
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(1 - depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_pkg["alpha"], os.path.join(depth_path, 'alpha_{0:05d}'.format(idx) + ".png"))

        depth_est = depth.squeeze().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2,0,1)
        torchvision.utils.save_image(depth_est, os.path.join(depth_path, 'color_{0:05d}'.format(idx) + ".png"))
    with open(os.path.join(model_path, name, 'render_time.txt'), 'w') as f:
        for t in render_time_list:
            f.write("%.2fms\n"%t)
        f.write("Mean time: %.2fms\n"%(np.mean(render_time_list[:])))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, near : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.dataset)
        (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
        gaussians.restore(model_params)
        gaussians.neural_renderer.keep_sigma=True

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "eval", scene.loaded_iter, scene.getEvalCameras(), gaussians, pipeline, background, near)
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--near", default=0, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.near)